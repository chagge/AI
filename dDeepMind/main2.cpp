#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <map>
#include <fstream>
#include <vector>
#include <string>

using namespace std;


struct epoch{
    int past, rw, acted, pres, isterm;
};

struct BInfo {
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

void saveStateFile(string x, int y) {
    string path = "savedState/";
    path += toString(y);
    ofstream myF(path.c_str());
    myF << x;
    myF.close();
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

    map<int, int> int2act, act2int;
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
    for(int i = 0; i < numActions; ++i)
        cout << int2act[i] << endl;

    int miniBatchSize = 32;

    int MILLION = 1000000;
    int i = 0;
    int maxIter = atoi(argv[2]), numFrameSkip = 4;
    double epsilon = 1.0, gamma = 0.8;
    ConvNet cnn(numActions);
    int expCount = 0;
    counter = 0;
    epoch *D = new epoch[MILLION];

    string garbage = inputString(out, 10);
    string frameStore = "", prevFrameStore = "";
    string input;
    input = "1,0,1,1\n";
    fwrite(input.c_str(), sizeof(char), input.length(), in);

    
    int toAct = rand()%(numActions);
    char *m[4];
    for(int kk = 0; kk < numFrameSkip-1; ++kk) {
        m[kk] = inputString(out, 10);
        frameStore += m[kk];
        string act = toString(int2act[toAct]) + ",18\n";
        fwrite(act.c_str(), sizeof(char), act.length(), in);
    }
    m[3] = inputString(out, 10);
    frameStore += m[3];
    saveStateFile(frameStore, counter);
    prevFrameStore.assign(frameStore);
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
            toAct = cnn.qArgMax(prevFrameStore);
        }
        
        cout << "i: " << i << "    " <<  "Action made: " << int2act[toAct] << endl;

        string act = toString(int2act[toAct]) + ",18\n";
        fwrite(act.c_str(), sizeof(char), act.length(), in);


        for(int kk = 0; kk < numFrameSkip-1; ++kk) {
            m[kk] = inputString(out, 10);
            frameStore += m[kk];
            string act = toString(int2act[toAct]) + ",18\n";
            fwrite(act.c_str(), sizeof(char), act.length(), in);
        }
        m[3] = inputString(out, 10);
        frameStore += m[3];

        if(!isTerminal(frameStore)) {
            saveStateFile(frameStore, counter);
            prevFrameStore.assign(frameStore);
            frameStore = "";

            D[expCount].past = counter-1;
            D[expCount].rw =  0;
            D[expCount].acted = toAct;
            D[expCount].pres = counter;
            D[expCount++].isterm = 0;
            counter = counter + 1;
        } else {
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
                y[kk] += cnn.qMax(D[tt].pres);
            }
            cnn.backPropDataPush(D[tt].acted, y[kk], D[tt].past);
        }
        cnn.backProp();

        i = i+4;
    }

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