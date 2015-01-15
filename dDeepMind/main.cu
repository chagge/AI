/*

Points to ponder:

-forward more optimization.
-Can we store the experience in some buffer if possible













*/


#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <map>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>

#define cudaMemcpyHTD(dest, src, nBytes) cudaMemcpy(dest, src, nBytes, cudaMemcpyHostToDevice)
#define cudaMemcpyDTH(dest, src, nBytes) cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToHost)

__device__ double recl(double x) {
    return log(1+exp(x));
}


__device__ double sqrtt(double x) {
    return sqrt(x);
}


__device__ double expp(double x) {
    return exp(x);
}

using namespace std;

#define MODUL 500
#define BLOCKSIZE 512

struct DIM4 {
    int x, y, z, w;
};

struct DIM2 {
    int x, y;
};

struct epoch{
    int past, rw, acted, pres, isterm;
};

struct BInfo {
    int actNum;
    double y;
    int fNum;
};

double rrgen() {
    return (double)(rand())/(double)(RAND_MAX);
}

class ConvNet {
    double *qVal;
    int numActions;
    vector<BInfo> *backPInfo;

public:
    ConvNet(int x) {
        numActions = x;
        qVal = new double[x];
        randInit();
        backPInfo = new vector<BInfo>[numActions];
    }
    ~ConvNet() {

    }
    void randInit() {
        for(int i = 0; i < numActions; ++i) {
            qVal[i] = rrgen();
        }
    }
    void forwardProp(int x) {
        if(x == -1) {
            for(int i = 0; i < numActions; ++i) {
                qVal[i] = rrgen();
            }
        } else {
            qVal[x] = rrgen();
        }
    }

    double qMax(int fileNum) {
        //preprocess fileNum and inp and store in double[] input
        //then feedforward once for each netowrk
        //all above in parallel
        double temp = qVal[0];
        for(int i = 0; i < numActions; ++i) {
            if(temp < qVal[i])
                temp = qVal[i];
        }
        return temp;
    }

    int qArgMax(string inp) {
        //preprocess inp and store in double[] input
        //then feedforward once for each netowrk
        //all above in parallel
        int temp = 0;
        for(int i = 0; i < numActions; ++i) {
            if(qVal[temp] < qVal[i])
                temp = i;
        }
        return temp;
    }

    void backProp() {
        //backpropagate using backPinfo Data and RMSProp
        for(int i = 0; i < numActions; ++i) {
            qVal[i] = rrgen();
        }
    }

    void backPropDataPush(int actNum, double output, int fileNum) {
        BInfo t = {output, fileNum};
        backPInfo[actNum].push_back(t);
    }
};

char *inputString(FILE* fp, size_t size){
//The size is extended by the input with the value of the provisional
    char *str;
    int ch;
    size_t len = 0;
    str = (char*)realloc(NULL, sizeof(char)*size);//size is start size
    if(!str)return str;
    while(EOF!=(ch=fgetc(fp)) && ch != '\n'){
        str[len++]=ch;
        if(len==size){
            str = (char*)realloc(str, sizeof(char)*(size+=16));
            if(!str)return str;
        }
    }
    str[len++]='\0';

    return (char*)realloc(str, sizeof(char)*len);
}

inline int toInt(string s) {int i;stringstream (s)>>i;return i;}
inline string toString(int i) {string s;stringstream ss;ss<<i;ss>>s;return s;}

int wtFileCounter = 0;
void saveWeights(double *wt, int num) {
    double *hwt = new double[num];
    cudaMemcpyDTH(hwt, wt, num*sizeof(double));
    string path = "savedWt/";
    path += toString(wtFileCounter);
    wtFileCounter++;
    ofstream myF(path.c_str());
    for(int i = 0; i < num; ++i) {
        myF << hwt[i] << endl;
    }
    myF.close();
    delete[] hwt;
}

string saveStateFile(string x, int y) {
    int ll = 160*210*2;
    string path = "savedState/";
    path += toString(y);
    string final = "", temp = "";
    stringstream ss;
    ss.str(x);
    while(ss >> temp) {
        final += temp.substr(0, ll);
    }
    ofstream myF(path.c_str());
    myF << final;
    myF.close();
    return final;
}

int isTerminal(string &x) {
    int count = 0;
    for(int i = x.size()-1; i>=0; --i) {
        if(x[i] == ':') {
            count++;
            if(count == 2) {
                if(x[i+1]=='1')
                    return true;
                else
                    return false;
            }
        }
        if(count == 2)
            return false;
    }
    return false;
}



__global__ void initWeights(double *in, int n) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx >= n)
        return;
    in[idx] = 0.24*(double)(idx%MODUL)/MODUL -0.12;
}

__global__ void forwardUtil(double *nni, double *nno, double *wt, int num, DIM4 *nnD, DIM4 *wtD, DIM2 *stride) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int dx = nnD[num+1].x;
    int dy = nnD[num+1].y;
    int dz = nnD[num+1].z;
    if(idx>=dx*dy*dz)
        return;
    int w_ = idx/(dx*dy);
    int j_ = (idx-w_*dx*dy)/dy;
    int k_ = idx - w_*dx*dy - j_*dy;
    double sum = 0;
    for(int i = 0; i < wtD[num].z; ++i) {
        for(int j = 0; j < wtD[num].x; ++j) {
            for(int k = 0; k < wtD[num].y; ++k) {
                sum += wt[w_*wtD[num].x*wtD[num].y*wtD[num].z+i*wtD[num].x*wtD[num].y+j*wtD[num].y+k]*nni[i*nnD[num].x*nnD[num].y+ (j+j_*stride[num].x)*nnD[num].y+ k+k_*stride[num].y];
            }
        }
    }
    nno[idx] = sum;
}

__global__ void findMax(double *nn, int n, int tp, double *maxT, DIM4 *nnD) {
    int idM = 0;
    for(int i = 0; i < nnD[n].z*nnD[n].y*nnD[n].x; ++i) {
        if(nn[i] > nn[idM])
            idM = i;
    }
    if(tp == 0)
        *maxT = idM;
    else
        *maxT = nn[idM];
}

double forwardProp(double *nn, double *wt, int num, DIM4 *nnD, DIM4 *wtD, DIM2 *stride, int type, DIM4 *hnnD, DIM4 *hwtD) {
    int prevBytes = 0, prevBytes2 = 0;
    for(int i = 0; i < num; ++i) {
        dim3 threadsPerBlock(BLOCKSIZE);
        dim3 numBlocks((hnnD[i+1].x*hnnD[i+1].y*hnnD[i+1].z-1)/threadsPerBlock.x + 1); 
        forwardUtil<<<numBlocks,threadsPerBlock>>>(nn + prevBytes, nn + prevBytes + hnnD[i].x*hnnD[i].y*hnnD[i].z, wt + prevBytes2, i, nnD, wtD, stride);
        prevBytes += hnnD[i].x*hnnD[i].y*hnnD[i].z;
        prevBytes2 += hwtD[i].x*hwtD[i].y*hwtD[i].z*hwtD[i].w;
    }
    double h_maxT, *d_maxT;
    cudaMalloc((void**)&d_maxT, sizeof(double));
    cudaMemset(d_maxT, 0, sizeof(double));
    findMax<<<1,1>>>(nn+prevBytes, num, type, d_maxT, nnD);
    cudaMemcpyDTH(&h_maxT, d_maxT, sizeof(double));
    return h_maxT;
}

__device__ double getVal(char c1, char c2) {
    double x = 0;
    if(c1 >= 'A' && c1 <= 'F') {
        x += 16*(c1 - 'A' + 10);
    } else {
        x += 16*(c1-'0');
    }
    if(c2 >= 'A' && c2 <= 'F') {
        x += (c2 - 'A' + 10);
    } else {
        x += (c2-'0');
    }
    return (x)/(16.0*16.0);
}

string preprocess(string x) {
    string y = "";
    int counter = 0;
    for(int t = 0; t < 4; ++t) {
        for(int i = 0; i < 210; i+=2) {
            if(i < 30 || i >= 198)
                continue;
            for(int j = 0; j < 160 + 4; ++j) {
                if(j >= 160) {
                    y += "00";
                }else {
                    y += x[counter++] + x[counter++];
                    counter+=2;
                }
            }
            counter += 160;
        }
    }
    return y;
}

__global__ void initFlayer(double *nn, char *x) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx >= 4*210*168)
        return;
    int t = idx / 4;
    int i = (idx - t*84*84)/84;
    int j = idx - t*84*84 - i*84;
    if(j>=80)
        nn[idx] = 0;
    else
        nn[idx] = getVal(x[t*2*210*160 + (2*i+30)*160 + 2*j], x[t*2*210*160 + (2*i+30)*160 + 2*j+1]);
}

double forwardPropMain(string x, double *nn, double *wt, int num, DIM4 *nnD, DIM4 *wtD, DIM2 *stride, int type, DIM4 *hnnD, DIM4 *hwtD) {
    double *fLayer = new double[hnnD[0].x*hnnD[0].y*hnnD[0].z];
    char *d_x;
    cudaMalloc((void**)&d_x, sizeof(char)*x.length());
    cudaMemcpyHTD(d_x, x.c_str(), sizeof(char)*x.length());
    dim3 threadsPerBlock(BLOCKSIZE);
    dim3 numBlocks((4*84*84-1)/threadsPerBlock.x + 1); 
    initFlayer<<<numBlocks, threadsPerBlock>>>(nn, d_x);
    //string y = "";
    /*
    int counter = 0;
    int cnt = 0;
    for(int t = 0; t < 4; ++t) {
        for(int i = 0; i < 210; i+=2) {
            if(i < 30 || i >= 198)
                continue;
            for(int j = 0; j < 160 + 8; j+=2) {
                if(j >= 160) {
                    //y += "00";
                    fLayer[cnt++] = 0;
                }else {
                    string ttt = x.substr(counter,2);
                    fLayer[cnt++] = getVal(ttt[0], ttt[1]);
                    //y += ttt;
                    counter+=4;
                }
            }
            counter += 160;
        }
    }
    */

    //string x = preprocess(x_);
    //cout << y.size() << "  " << hnnD[0].x*hnnD[0].y*hnnD[0].z*2 << " " << cnt << endl;
    //cudaMemcpyHTD(nn, fLayer, hnnD[0].x*hnnD[0].y*hnnD[0].z*sizeof(double));
    return forwardProp(nn, wt, num, nnD, wtD, stride, type, hnnD, hwtD);
}

double forwardPropFileMain(int x, double *nn, double *wt, int num, DIM4 *nnD, DIM4 *wtD, DIM2 *stride, int type, DIM4 *hnnD, DIM4 *hwtD) {
    ifstream ifs(("savedState/"+toString(x)).c_str());
    std::string content((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
    ifs.close();
    return forwardPropMain(content, nn, wt, num, nnD, wtD, stride, type, hnnD, hwtD);
}

vector<BInfo> backPropData;

__global__ void setFinalLayer(double *nn, double *err, int a, double y) {
    err[a] = nn[a] - y;
}

void backPropDataPush(int actN, double output, int input) {
    BInfo bb={actN, output, input};
    backPropData.push_back(bb);
}

__device__ double derivv(double x) {
    return (1.0 - 1.0/expp(x));
}

__global__ void updateWts(double *err, double *nn, double *wt, DIM4 *nnD, DIM4 *wtD, DIM2 *stride, int num, double *msq, int maxNum) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx >= maxNum)
        return;
    int w = idx / (wtD[num].x*wtD[num].y*wtD[num].z);
    int c = (idx - w*wtD[num].x*wtD[num].y*wtD[num].z)/(wtD[num].x*wtD[num].y);
    int b = (idx - w*wtD[num].x*wtD[num].y*wtD[num].z - c * wtD[num].x*wtD[num].y) / (wtD[num].y);
    int a = idx - w*wtD[num].x*wtD[num].y*wtD[num].z - c * wtD[num].x*wtD[num].y - b*wtD[num].y;
    double alpha = 0.3;
    double sum = 0;

    for(int i = 0; i < nnD[num+1].x; ++i) {
        for(int j = 0; j < nnD[num+1].y; ++j) {
            sum += err[w*nnD[num+1].x*nnD[num+1].y + i*nnD[num+1].y + j]*derivv(nn[w*nnD[num+1].x*nnD[num+1].y + i*nnD[num+1].y + j])*nn[c*nnD[num].x*nnD[num].y + (i*stride[num].x + a)*nnD[num].y + j*stride[num].y+b];
        }
    }
    msq[idx] = msq[idx]*0.9 + sum*sum*0.1;
    wt[idx] = wt[idx] + (sum*alpha)/sqrtt(msq[idx]);
}

__global__ void propagateError(double *errI, double *errO, double *nn, double *wt, DIM4 *nnD, DIM4 *wtD, DIM2 *stride, int num, int maxNum) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx >= maxNum)
        return;
    int k = idx / (nnD[num].x*nnD[num].y);
    int i = (idx - k*nnD[num].x*nnD[num].y)/nnD[num].y;
    int j = idx - k*nnD[num].x*nnD[num].y - i*nnD[num].y;
    double sum = 0;
    for(int u = 0; u < wtD[num].w; ++u) {
        for(int a = i%stride[num].x; a < wtD[num].x && i - a >= 0; a+=stride[num].x) {
            for(int b = j%stride[num].y; b < wtD[num].y && j - b >= 0; b+=stride[num].y) {
                sum += wt[u*wtD[num].x*wtD[num].y*wtD[num].z + k*wtD[num].x*wtD[num].y + a*wtD[num].y +b]*errI[u*nnD[num+1].x*nnD[num+1].y+(i-a)*nnD[num+1].y+(j-b)]*derivv(nn[u*nnD[num+1].x*nnD[num+1].y+(i-a)*nnD[num+1].y+(j-b)]);
            }
        }
    }
    errO[idx] = sum;
}

void backPropMain(double *nn, double *wt, int num,  DIM4 *nnD, DIM4 *wtD, DIM2 *stride, int type, DIM4 *hnnD, DIM4 *hwtD, int nnNumUnits, int wtNumUnits) {
    double *d_err, *d_msq;
    cudaMalloc((void**)&d_err, nnNumUnits*sizeof(double));
    cudaMalloc((void**)&d_msq, wtNumUnits*sizeof(double));
    cudaMemset((void**)&d_msq, 0, wtNumUnits*sizeof(double));
    for(int i = 0; i < backPropData.size(); ++i) {
        cudaMemset((void**)&d_err, 0, nnNumUnits*sizeof(double));
        double garbage = forwardPropFileMain(backPropData[i].fNum, nn, wt, num, nnD, wtD, stride, type, hnnD, hwtD);
        int lBytes = nnNumUnits - hnnD[num].x*hnnD[num].y*hnnD[num].z;
        int wtLBytes = wtNumUnits;
        setFinalLayer<<<1,1>>>(nn + lBytes, d_err + lBytes, backPropData[i].actNum, backPropData[i].y);
        for(int j = num-1; j >= 0; --j) {
            wtLBytes -= hwtD[j].x*hwtD[j].y*hwtD[j].z*hwtD[j].w;
            dim3 threadsPerBlock(BLOCKSIZE);
            dim3 numBlocks((hwtD[j].x*hwtD[j].y*hwtD[j].z*hwtD[j].w-1)/threadsPerBlock.x + 1); 
            dim3 numBlocks2((hnnD[j].x*hnnD[j].y*hnnD[j].z*hnnD[j].w-1)/threadsPerBlock.x + 1); 


            propagateError<<<numBlocks2, threadsPerBlock>>>(d_err + lBytes, d_err + lBytes - hnnD[j].x*hnnD[j].y*hnnD[j].z, nn + lBytes, wt + wtLBytes, nnD, wtD, stride, j, hnnD[j].x*hnnD[j].y*hnnD[j].z*hnnD[j].w);


            updateWts<<<numBlocks, threadsPerBlock>>>(d_err + lBytes, nn + lBytes - hnnD[j].x*hnnD[j].y*hnnD[j].z, wt + wtLBytes, nnD, wtD, stride, j, d_msq, hwtD[j].x*hwtD[j].y*hwtD[j].z*hwtD[j].w);
            lBytes -= hnnD[j].x*hnnD[j].y*hnnD[j].z;
        }
    }
    backPropData.clear();
    cudaFree(d_err);
    cudaFree(d_msq);
}


double getReward(string x) {
    //reward..set
    int ll=210*160*2;
    stringstream ss;
    string temp;
    ss.str(x);
    int rr = 0;
    int frew = 0;
    while(ss >> temp) {
        rr = atoi(temp.substr(ll+3, -1).c_str());
        //cout << temp.substr(ll+3, -1) << endl;
        if(rr > 0)
            frew = 1;
        else if(rr == 0)
            frew = 0;
        else
            frew = -1;
    }
    return frew;
}



int main(int argc, char **argv) {
    FILE *in,*out;
    if(!(in = popen("cat > ale_fifo_in", "w")))
    {
        return 0;
    }
    if(!(out = popen("cat ale_fifo_out", "r")))
    {
        return 0;
    }
    setbuf(in, NULL);
    setbuf(out, NULL);

    /*************************************************************/
    /*****************Decoding config file of Rom ****************/
    /*************************************************************/

    map<int, int> int2act;
    int resetButton;
    int numActions;
    int counter = 0;
    ifstream configF(argv[1]);
    if(configF.good()) {
        string temp;
        while(getline(configF, temp)) {
            if(temp != "") {
                if(counter == 0) {
                    numActions = atoi(temp.c_str());
                } else if(counter == numActions+1) {
                    resetButton = atoi(temp.c_str());
                } else {
                    int2act[counter-1] = atoi(temp.c_str());
                }
                counter++;
            }
        }
    }
    configF.close();

    /*************************************************************/
    /*****************Decoding Ends here *************************/
    /*************************************************************/



    /************************************************************************/
    /************************************************************************/
    //Convolution Neural Net initialization
    /************************************************************************/
    /************************************************************************/

    ifstream nnConfig(argv[2]);
    double *d_nnLayer;
    double *d_wtLayer;
    
    int nnNumUnits = 0, wtNumUnits = 0;

    int numNN, numWT;
    nnConfig >> numNN;
    numWT = numNN - 1;

    DIM4 *h_nnD = new DIM4[numNN];
    DIM4 *h_wtD = new DIM4[numWT];
    DIM2 *h_stride = new DIM2[numWT];
    DIM4 *d_nnD;
    DIM4 *d_wtD;
    DIM2 *d_stride;

    nnConfig >> h_nnD[0].z >> h_nnD[0].x >> h_nnD[0].y;
    h_nnD[0].w = 1;

    for(uint i = 0; i < numWT; ++i) {
        nnConfig >> h_wtD[i].w >> h_wtD[i].x >> h_wtD[i].y >> h_wtD[i].z;
        nnConfig >> h_stride[i].x >> h_stride[i].y;
        
        h_nnD[i+1].w = 1;
        h_nnD[i+1].z = h_wtD[i].w;
        h_nnD[i+1].y = (h_nnD[i].y - h_wtD[i].y + h_stride[i].y) / h_stride[i].y;
        h_nnD[i+1].x = (h_nnD[i].x - h_wtD[i].x + h_stride[i].x) / h_stride[i].x;
    }
    
    double alpha;
    nnConfig >> alpha;
    

    for(int i = 0; i < numNN; ++i)
        nnNumUnits += h_nnD[i].x * h_nnD[i].y * h_nnD[i].z;
    for(int i = 0; i < numWT; ++i)
        wtNumUnits += h_wtD[i].x * h_wtD[i].y * h_wtD[i].z * h_wtD[i].w;

    cudaMalloc((void**)&d_nnLayer, nnNumUnits*sizeof(double));
    cudaMalloc((void**)&d_wtLayer, wtNumUnits*sizeof(double));
    cudaMalloc((void**)&d_nnD, numNN*sizeof(DIM4));
    cudaMalloc((void**)&d_wtD, numWT*sizeof(DIM4));
    cudaMalloc((void**)&d_stride, numWT*sizeof(DIM2));


    cudaMemcpyHTD(d_nnD, h_nnD, numNN*sizeof(DIM4));
    cudaMemcpyHTD(d_wtD, h_wtD, numWT*sizeof(DIM2));

    dim3 threadsPerBlock(BLOCKSIZE);
    dim3 numBlocks((wtNumUnits-1)/threadsPerBlock.x + 1); 

    cout << "weights initialization started!" << endl;

    initWeights<<<numBlocks, threadsPerBlock>>>(d_wtLayer, wtNumUnits);

    cout << "weights initialized!" << endl;

    #ifdef DEB
        double *nndeb, *wtdeb;
        nndeb = new double[nnNumUnits];
        wtdeb = new double[wtNumUnits];
        cudaMemset(d_nnLayer, 1, nnNumUnits*sizeof(double));

        double x = forwardProp(d_nnLayer, d_wtLayer, numWT, d_nnD, d_wtD, d_stride, 0, h_nnD, h_wtD);

        cudaMemcpyDTH(nndeb, d_nnLayer, nnNumUnits*sizeof(double));
        cudaMemcpyDTH(wtdeb, d_wtLayer, wtNumUnits*sizeof(double));

        for(int i = 0; i < nnNumUnits; ++i)
            cout << nndeb[i] << endl;
        for(int i = 0; i < wtNumUnits; ++i)
            cout << wtdeb[i] << endl;

        cout << "max :" << x << endl;
        exit(1)
    #endif


    /*random try
    for(int i = 0; i < h_nnD[0].x*h_nnD[0].y*h_nnD[0].z; ++i)
        h_firstNNLayer[i] = (double)rand()/(double)RAND_MAX;

    cudaMemcpyHTD(d_nnLayer, h_firstNNLayer,h_nnD[0].x*h_nnD[0].y*h_nnD[0].z*sizeof(double));

    double x = forwardProp(d_nnLayer, d_wtLayer, numWT, d_nnD, d_wtD, d_stride, 0, h_nnD, h_wtD);

    cout << x << endl;
    */


    /************************************************************************/
    // Convolution Neural Net init ends here..................................
    /************************************************************************/


    /************************************************************************/
    // Q learning variable initialization
    /************************************************************************/


    int miniBatchSize = 32;
    int MILLION = 1000000;
    int i = 0;
    int maxIter = atoi(argv[3]), numFrameSkip = 4;
    double epsilon = 1.0;
    int expCount = 0;
    counter = 0;
    epoch *D = new epoch[MILLION];

    string garbage = inputString(out, 10);
    string frameStore = "", prevFrameStore = "";
    string input;
    input = "1,0,1,1\n";
    fwrite(input.c_str(), sizeof(char), input.length(), in);

    /*************************************************************************/
    // Qlearning variable init ends here
    /*************************************************************************/



    /*************************************************************************/
    /*************************************************************************/
    /*************************************************************************/
    /******* *******      ***      **     ***  **** ** **  **** **      ******/
    /******* ******* ******** **** ** *** *** * *** ** ** * *** ** ***********/
    /******* *******    *****      **    **** ** ** ** ** ** ** ** **   ******/
    /******* ******* ******** **** ** *** *** *** * ** ** *** * ** **** ******/
    /*******      **      *** **** ** **** ** ****  ** ** ****  **      ******/
    /*************************************************************************/
    /*************************************************************************/


    int toAct = rand()%(numActions);
    char *m[4];
    for(int kk = 0; kk < numFrameSkip-1; ++kk) {
        m[kk] = inputString(out, 10);
        frameStore += m[kk];
        frameStore += "\n";
        string act = toString(int2act[toAct]) + ",18\n";
        fwrite(act.c_str(), sizeof(char), act.length(), in);
    }
    m[3] = inputString(out, 10);
    frameStore += m[3];
    prevFrameStore = saveStateFile(frameStore, counter%MILLION);
    frameStore = "";

    D[expCount].past = -1;
    D[expCount].rw =  0;
    D[expCount].acted = toAct;
    D[expCount].pres = counter;
    D[expCount++].isterm = 0;
    counter = counter + 1;


    i = i + 4;
    while(i < maxIter - numFrameSkip) {

        double unif = (double)(rand())/(double)(RAND_MAX);
        if(unif < epsilon) {
            toAct = rand()%(numActions);
        } else {
            toAct = forwardPropMain(prevFrameStore, d_nnLayer, d_wtLayer, numWT, d_nnD, d_wtD, d_stride, 0, h_nnD, h_wtD);
        }
        
        cout << "i: " << i << "    " <<  "Action made: " << int2act[toAct] << endl;

        string act = toString(int2act[toAct]) + ",18\n";
        fwrite(act.c_str(), sizeof(char), act.length(), in);


        for(int kk = 0; kk < numFrameSkip-1; ++kk) {
            m[kk] = inputString(out, 10);
            frameStore += m[kk];
            frameStore += "\n";
            string act = toString(int2act[toAct]) + ",18\n";
            fwrite(act.c_str(), sizeof(char), act.length(), in);
        }
        m[3] = inputString(out, 10);
        frameStore += m[3];

        if(!isTerminal(frameStore)) {
            prevFrameStore = saveStateFile(frameStore, counter%MILLION);
            D[expCount].past = (counter-1)%MILLION;
            D[expCount].rw =  getReward(frameStore);
            D[expCount].acted = toAct;
            D[expCount].pres = (counter)%MILLION;
            D[expCount++].isterm = 0;
            counter = counter + 1;
            frameStore = "";
        } else {
            //cout << "TERMINAL" << endl;
            prevFrameStore = saveStateFile(frameStore, counter%MILLION);
            D[expCount].past = (counter-1)%MILLION;
            D[expCount].rw =  -1;
            D[expCount].acted = toAct;
            D[expCount].pres = (counter)%MILLION;
            D[expCount++].isterm = 1;
            counter = counter + 1;
            frameStore = "";
            string act = toString(resetButton) + ",18\n";
            fwrite(act.c_str(), sizeof(char), act.length(), in);
            string garbage = inputString(out, 10);
        }


        double *y = new double[miniBatchSize];
        for(int kk = 0; kk < miniBatchSize; ++kk) {
            int tt = rand()%(min(MILLION-1, expCount-1)) + 1;
            y[kk] = D[tt].rw;
            if(D[tt].isterm==0) {
                y[kk] += forwardPropFileMain(D[tt].pres, d_nnLayer, d_wtLayer, numWT, d_nnD, d_wtD, d_stride, 1, h_nnD, h_wtD);
                cout << y[kk] - D[tt].rw << "  reward " << endl;
                //y[kk] += cnn.qMax(D[tt].pres);
            }
            backPropDataPush(D[tt].acted, y[kk], D[tt].past);
        }
        backPropMain(d_nnLayer, d_wtLayer, numWT, d_nnD, d_wtD, d_stride, 0, h_nnD, h_wtD, nnNumUnits, wtNumUnits);
        if(i%1000 == 0)
            saveWeights(d_wtLayer, wtNumUnits);
        i = i+4;
        if(i >= MILLION) {
            epsilon = 0.1;
        } else
            epsilon = 1.0 - (double)i / (double)maxIter;
    }
    saveWeights(d_wtLayer, wtNumUnits);
    while(1) {
        string act =  "0,18\n";
        fwrite(act.c_str(), sizeof(char), act.length(), in);
        char *dieString = inputString(out, 10);
        if(dieString[0] == 'D') {
            cout << "DIE" << endl;
            break;
        }
    }
}

//./ale -player_agent keyboard_agent -max_num_frames 10 -game_controller fifo -run_length_encoding false /home/dhruv/Roms/ROMS/amidar.bin