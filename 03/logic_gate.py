import numpy as np
class LogicGate:
    def __init__ (self):
        self.w1 = None
        self.w2 = None
        self.th = None
        self.out = None
        self.x1 = None
        self.x2 = None

    def printOutput(self,gateVal):
        if(gateVal == 'AND'):
            print(f'{self.x1} AND {self.x2} = {self.out}')
        elif(gateVal == 'OR'):
            print(f'{self.x1} OR {self.x2} = {self.out}')
        elif(gateVal == 'NAND'):
            print(f'{self.x1} NAND {self.x2} = {self.out}')
        elif(gateVal == 'NOR'):
            print(f'{self.x1} NOR {self.x2} = {self.out}')
        elif(gateVal == 'XOR'):
            print(f'{self.x1} XOR {self.x2} = {self.out}')
    
    def andGate(self,x1,x2):
        self.w1 = 1
        self.w2 = 1
        self.th = 2
        self.x1 = x1
        self.x2 = x2
        x = np.array([self.x1,self.x2])
        w = np.array([self.w1,self.w2])

        if(np.sum(x*w)>=self.th):
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0
        
    def orGate(self,x1,x2):
        self.w1 = 1
        self.w2 = 1
        self.th = 1
        self.x1 = x1
        self.x2 = x2
        x = np.array([self.x1,self.x2])
        w = np.array([self.w1,self.w2])

        if(np.sum(x*w)>=self.th):
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0
        
    def nandGate(self,x1,x2):
        self.w1 = -1
        self.w2 = -1
        self.th = -1
        self.x1 = x1
        self.x2 = x2
        x = np.array([self.x1,self.x2])
        w = np.array([self.w1,self.w2])

        if(np.sum(x*w)>=self.th):
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0
        
    def norGate(self,x1,x2):
        self.w1 = -1
        self.w2 = -1
        self.th = 0
        self.x1 = x1
        self.x2 = x2
        x = np.array([self.x1,self.x2])
        w = np.array([self.w1,self.w2])

        if(np.sum(x*w)>=self.th):
            self.out = 1
            return 1
        else:
            self.out = 0
            return 0


    # XOR Gate is more complex than the basic gates so we use multiple basic gates to implement this.     
    def xorGate(self,x1,x2):
        self.x1 = x1
        self.x2 = x2

        #XOR(A,B) = (A OR B) AND (A NAND B)
        orGateResult = self.orGate(self.x1,self.x2)
        nandGateResult = self.nandGate(self.x1,self.x2)

        xorResultVal = self.andGate(orGateResult,nandGateResult)

        self.out = xorResultVal
        return xorResultVal
        
        
if(__name__ == '__main__'):
    print("Hello World, This is the basic logicGates101.")
    print("You can use this file to learn Basic logic gates like: AND, OR, NAND, NOR, and XOR")

    # Modify the below code to change the gates and inputs.
    #Sample usage
    logicGate = LogicGate()
    logicGate.andGate(1,1)
    logicGate.printOutput('AND')