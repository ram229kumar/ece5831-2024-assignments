import logic_gate as lg

logicGate = lg.LogicGate()


# Testing for AND Gate
print('--------------------------------------')

logicGate.andGate(0,0)
logicGate.printOutput('AND')
logicGate.andGate(0,1)
logicGate.printOutput('AND')
logicGate.andGate(1,0)
logicGate.printOutput('AND')
logicGate.andGate(1,1)
logicGate.printOutput('AND')
print('--------------------------------------')

# Testing for OR Gate

logicGate.orGate(0,0)
logicGate.printOutput('OR')
logicGate.orGate(0,1)
logicGate.printOutput('OR')
logicGate.orGate(1,0)
logicGate.printOutput('OR')
logicGate.orGate(1,1)
logicGate.printOutput('OR')
print('--------------------------------------')

# Testing for NAND Gate

logicGate.nandGate(0,0)
logicGate.printOutput('NAND')
logicGate.nandGate(0,1)
logicGate.printOutput('NAND')
logicGate.nandGate(1,0)
logicGate.printOutput('NAND')
logicGate.nandGate(1,1)
logicGate.printOutput('NAND')
print('--------------------------------------')

# Testing for NOR Gate

logicGate.norGate(0,0)
logicGate.printOutput('NOR')
logicGate.norGate(0,1)
logicGate.printOutput('NOR')
logicGate.norGate(1,0)
logicGate.printOutput('NOR')
logicGate.norGate(1,1)
logicGate.printOutput('NOR')
print('--------------------------------------')

# Testing for XOR Gate

logicGate.xorGate(0,0)
logicGate.printOutput('XOR')
logicGate.xorGate(0,1)
logicGate.printOutput('XOR')
logicGate.xorGate(1,0)
logicGate.printOutput('XOR')
logicGate.xorGate(1,1)
logicGate.printOutput('XOR')
print('--------------------------------------')