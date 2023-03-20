
import numpy as np

def GenerateTrainingData_2DFeature(min,max):
    '''GenerateTrainingData: example of use: DesignMatrix,TrainingValues= GenerateTrainingData(1,11)
    generates data points from 1 to 10 in steps of 1 and uses them to calculate dependent values using 
    the function defined in ModelFuction()'''
    DesignMatrixString1=[]  # Generate as string and then convert, below, to array 
    DesignMatrixString2=[]
    TrainingValues=[]
    for data in range(min,max):
        DesignMatrixString1.append(data)
        DesignMatrixString2.append(data)
    DesignMatrix1 = np.asarray(DesignMatrixString1).reshape((1,(max-min)))
    DesignMatrix2 = np.asarray(DesignMatrixString2).reshape((1,(max-min)))
    TrainingValues = np.asarray(ModelFunction2D(DesignMatrix1, DesignMatrix2)).reshape((1,(max-min)))
    #DesignMatrix=np.asarray(DesignMatrixString)
    #TrainingValues=np.asarray(ModelFunction(DesignMatrix))
    return(DesignMatrix1, DesignMatrix2, TrainingValues)  

def ModelFunction2D(DesignMatrix1, DesignMatrix2):
    return(5.0+DesignMatrix1*3.0 + DesignMatrix2*5.0)


DesignMatrix1, DesignMatrix2, TrainingValues = GenerateTrainingData_2DFeature(1,11)
print(DesignMatrix1, DesignMatrix2, TrainingValues)