#Author Haris Prasovi 


import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LinearRegression

# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("train.csv")
    testDF = pd.read_csv("test.csv")

    #demonstrateHelpers(trainDF)

    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    
    doExperiment(trainInput, trainOutput, predictors)
    
    ##doRidgeReg(trainInput, trainOutput, predictors)
    
    #doLassoRef(trainInput, trainOutput, predictors)
    
    #doLassoLarsRef(trainInput, trainOutput, predictors)
    
    doBayesianRidge(trainInput, trainOutput, predictors)
    
    #dorand(trainInput, trainOutput, predictors)
    
    doGradBoost(trainInput, trainOutput, predictors)
    

    
    #doLogisticRidge(trainInput, trainOutput, predictors)
    
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)

    
# ===============================================================================
'''
Does k-fold CV on the Kaggle training set using LinearRegression.
'''

#GradBoost(more derivative=beter results)
def doGradBoost(trainInput, trainOutput, predictors):
    alg = GradientBoostingRegressor()
    alg.fit(trainInput.loc[:, predictors], trainOutput)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score for GradientBoosting Regressor:", cvMeanScore)

'''
def dorand(trainInput, trainOutput, predictors):
    alg = GradientBoostingRegressor()
    alg.fit(trainInput.loc[:, predictors], trainOutput)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score for rand Regressor:", cvMeanScore)
'''

#Original privded Linear Regression Model
def doExperiment(trainInput, trainOutput, predictors):
    alg = LinearRegression()
    alg.fit(trainInput.loc[:, predictors], trainOutput)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score for Linear Regression:", cvMeanScore)


#Ridge Model(not as good with our data)
def doRidgeReg(trainInput, trainOutput, predictors):
    alg=linear_model.Ridge(alpha=.5)
    
    alg.fit(trainInput.loc[:, predictors], trainOutput)
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()

    print("CV Average Score for Ridge Regression:", cvMeanScore)
    
#Lasso Ridge(Weird comment about length of y input)
def doLassoRef(trainInput, trainOutput, predictors):
    alg=linear_model.Lasso(alpha=0.1)
    
    alg.fit(trainInput.loc[:, predictors], trainOutput)
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()

    print("CV Average Score for Lasso Regression:", cvMeanScore)
    
#LassoLars(weir comment as well)  
def doLassoLarsRef(trainInput, trainOutput, predictors):
    alg=linear_model.LassoLars(alpha=0.1)
    
    alg.fit(trainInput.loc[:, predictors], trainOutput)
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()

    print("CV Average Score for LassoLars Regression:", cvMeanScore)
    
#BayesianRidge(Works well with weird data)
def doBayesianRidge(trainInput, trainOutput, predictors):
    alg=linear_model.BayesianRidge()
    
    alg.fit(trainInput.loc[:, predictors], trainOutput)
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()

    print("CV Average Score for BayesianRidge Regression:", cvMeanScore)
  
'''
def doLogisticRidge(trainInput, trainOutput, predictors):
    alg=linear_model.LogisticRegression()
    
    alg.fit(trainInput.loc[:, predictors], trainOutput)
    
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()

    print("CV Average Score for Logistic Regression:", cvMeanScore)
 '''   
    
# ===============================================================================
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
    alg = LinearRegression()

    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle

# ============================================================================
# Data cleaning - conversion, normalization

'''
Pre-processing code 
'''
def transformData(trainDF, testDF):
 
    predictors = ['YearBuilt','YearRemodAdd','Neighborhood','Electrical','PavedDrive','PoolQC','PoolArea','MiscVal','KitchenQual','GarageCars','GarageCond','Fence','GarageQual','BsmtExposure','TotalBsmtSF','HeatingQC','Utilities','GrLivArea','KitchenAbvGr','BedroomAbvGr','TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','LotArea','TotalBsmtSF','GrLivArea','MasVnrArea',"Functional",'Alley',"CentralAir",'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',"GarageArea","LotFrontage",'1stFlrSF', '2ndFlrSF',"ExterQual","BsmtCond","ExterCond","TotRmsAbvGrd","OverallQual","OverallCond"]
    #predictor = ['Neighborhood','Electrical','PavedDrive','PoolQC','PoolArea','MiscVal','KitchenQual','GarageCars','GarageCond','Fence','GarageQual','BsmtExposure','TotalBsmtSF','HeatingQC','Utilities','GrLivArea','KitchenAbvGr','BedroomAbvGr','TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','LotArea','TotalBsmtSF','GrLivArea','MasVnrArea',"Functional",'Alley',"CentralAir",'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',"GarageArea","LotFrontage",'1stFlrSF', '2ndFlrSF',"ExterQual","BsmtCond","ExterCond","TotRmsAbvGrd","OverallQual","OverallCond"]
   
    print(len(predictors), " out of : ", len(trainDF.columns))
   
    
    trainInput = trainDF.loc[:, predictors]
    testInput = testDF.loc[:, predictors]
    
    #Replace all NaN with mean of the column
    inp=['LotArea','1stFlrSF','MasVnrArea',"LotFrontage"]
    replacewmean(trainInput, inp)
    replacewmean(testInput, inp)
    
    #Change Letters to numbers for Y?N
    testInput.loc[:, "CentralAir"] = testInput.loc[:, "CentralAir"].map(lambda v: 0 if v=="N" else 1)
    trainInput.loc[:, "CentralAir"] = trainInput.loc[:, "CentralAir"].map(lambda v: 0 if v=="N" else 1)

    
  
    
    
    #All the data that needed numerical values 
    testInput.loc[:, 'PavedDrive'] = testInput.loc[:, 'PavedDrive'].map(lambda row:1 if row=='P' else(2 if row=='Y' else 0))
    trainInput.loc[:, 'PavedDrive'] = trainInput.loc[:, 'PavedDrive'].map(lambda row:1 if row=='P' else(2 if row=='Y' else 0))

    
    testInput.loc[:, 'Alley'] = testInput.loc[:, 'Alley'].map(lambda row:1 if row=='Grvl' else(2 if row=='Pave' else 0))
    trainInput.loc[:, 'Alley'] = trainInput.loc[:, 'Alley'].map(lambda row:1 if row=='Grvl' else(2 if row=='Pave' else 0))

    #replace NaN with the mode
    repwmo=['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']
    replacewmode(trainInput,repwmo)
    replacewmode(testInput,repwmo)
   
    testInput.loc[:,'BsmtCond']= testInput.loc[:,'BsmtCond'].map(lambda row: 0 if row=='Po' else(1 if row=='Fa' else (2 if row=='TA' else(3 if row=='Gd' else 4))))
    trainInput.loc[:,'BsmtCond']= trainInput.loc[:,'BsmtCond'].map(lambda row: 0 if row=='Po' else(1 if row=='Fa' else (2 if row=='TA' else(3 if row=='Gd' else 4))))
    
    testInput.loc[:,'Fence']= testInput.loc[:,'Fence'].map(lambda row: 2 if row=='NA' else(1 if row=='MnWw' else (2 if row=='GdWo' else(3 if row=='MnPrv' else 4))))
    trainInput.loc[:,'Fence']= trainInput.loc[:,'Fence'].map(lambda row: 2 if row=='NA' else(1 if row=='MnWw' else (2 if row=='GdWo' else(3 if row=='MnPrv' else 4))))
   
    testInput.loc[:,'ExterCond']= testInput.loc[:,'ExterCond'].map(lambda row: 5 if row=='Ex' else(4 if row=='Gd' else (3 if row=='TA' else(2 if row=='Fa' else(1 if row=='Po' else 0)))))
    trainInput.loc[:,'ExterCond']= trainInput.loc[:,'ExterCond'].map(lambda row: 5 if row=='Ex' else(4 if row=='Gd' else (3 if row=='TA' else(2 if row=='Fa' else(1 if row=='Po' else 0)))))

    testInput.loc[:,'PoolQC']= testInput.loc[:,'PoolQC'].map(lambda row: 5 if row=='Ex' else(4 if row=='Gd' else (3 if row=='TA' else(2 if row=='Fa' else(1 if row=='Po' else 0)))))
    trainInput.loc[:,'PoolQC']= trainInput.loc[:,'PoolQC'].map(lambda row: 5 if row=='Ex' else(4 if row=='Gd' else (3 if row=='TA' else(2 if row=='Fa' else(1 if row=='Po' else 0)))))

    testInput.loc[:,'KitchenQual']= testInput.loc[:,'KitchenQual'].map(lambda row: 5 if row=='Ex' else(4 if row=='Gd' else (3 if row=='TA' else(2 if row=='Fa' else(1 if row=='Po' else 0)))))
    trainInput.loc[:,'KitchenQual']= trainInput.loc[:,'KitchenQual'].map(lambda row: 5 if row=='Ex' else(4 if row=='Gd' else (3 if row=='TA' else(2 if row=='Fa' else(1 if row=='Po' else 0)))))

    
    testInput.loc[:,'GarageQual']= testInput.loc[:,'GarageQual'].map(lambda row: 5 if row=='Ex' else(4 if row=='Gd' else (3 if row=='TA' else(2 if row=='Fa' else(1 if row=='Po' else 0)))))
    trainInput.loc[:,'GarageQual']= trainInput.loc[:,'GarageQual'].map(lambda row: 5 if row=='Ex' else(4 if row=='Gd' else (3 if row=='TA' else(2 if row=='Fa' else(1 if row=='Po' else 0)))))

    testInput.loc[:,'GarageCond']= testInput.loc[:,'GarageCond'].map(lambda row: 5 if row=='Ex' else(4 if row=='Gd' else (3 if row=='TA' else(2 if row=='Fa' else(1 if row=='Po' else(0 if row=='NA' else 2))))))
    trainInput.loc[:,'GarageCond']= trainInput.loc[:,'GarageCond'].map(lambda row: 5 if row=='Ex' else(4 if row=='Gd' else (3 if row=='TA' else(2 if row=='Fa' else(1 if row=='Po' else(0 if row=='NA' else 2))))))


    testInput.loc[:,'HeatingQC']= testInput.loc[:,'HeatingQC'].map(lambda row: 5 if row=='Ex' else(4 if row=='Gd' else (3 if row=='TA' else(2 if row=='Fa' else(1 if row=='Po' else(0 if row=='NA' else 2))))))
    trainInput.loc[:,'HeatingQC']= trainInput.loc[:,'HeatingQC'].map(lambda row: 5 if row=='Ex' else(4 if row=='Gd' else (3 if row=='TA' else(2 if row=='Fa' else(1 if row=='Po' else(0 if row=='NA' else 2))))))
    
    
    testInput.loc[:,'ExterQual']= testInput.loc[:,'ExterQual'].map(lambda row: 0 if row=='Po' else(1 if row=='Fa' else (2 if row=='TA' else 3)))
    trainInput.loc[:,'ExterQual']= trainInput.loc[:,'ExterQual'].map(lambda row: 0 if row=='Po' else(1 if row=='Fa' else (2 if row=='TA' else 3)))
    
    testInput.loc[:,'BsmtExposure']= testInput.loc[:,'BsmtExposure'].map(lambda row: 4 if row=='Gd' else (3 if row=='Av' else(2 if row=='Mn' else(1 if row=='' else 0))))
    testInput.loc[:,'BsmtExposure']= testInput.loc[:,'BsmtExposure'].map(lambda row: 4 if row=='Gd' else (3 if row=='TA' else(2 if row=='Fa' else(1 if row=='Po' else 0))))
    
    
    testInput.loc[:,'ExterQual']= testInput.loc[:,'ExterQual'].map(lambda row: 4 if row=='Gd' else (3 if row=='TA' else(2 if row=='Fa' else(1 if row=='Po' else 0))))
    testInput.loc[:,'ExterQual']= testInput.loc[:,'ExterQual'].map(lambda row: 4 if row=='Gd' else (3 if row=='TA' else(2 if row=='Fa' else(1 if row=='Po' else 0))))
    
    
    testInput.loc[:,'Utilities']= testInput.loc[:,'Utilities'].map(lambda row: 4 if row=='AllPub' else(3 if row=='NoSewr' else (2 if row=='NoSeWa' else(1 if row=='ELO' else 0))))
    trainInput.loc[:,'Utilities']= trainInput.loc[:,'Utilities'].map(lambda row: 4 if row=='AllPub' else(3 if row=='NoSewr' else (2 if row=='NoSeWa' else(1 if row=='ELO' else 0))))

    testInput.loc[:,'Electrical']= testInput.loc[:,'Electrical'].map(lambda row: 4 if row=='SBrkr' else(3 if row=='FuseA' else (2 if row=='FuseF' else(1 if row=='FuseP' else 0))))
    trainInput.loc[:,'Electrical']= trainInput.loc[:,'Electrical'].map(lambda row: 4 if row=='SBrkr' else(3 if row=='FuseA' else (2 if row=='FuseF' else(1 if row=='FuseP' else 0))))


    testInput.loc[:,'Functional']= testInput.loc[:,'Functional'].map(lambda row: 0 if row=='Sal' else(1 if row=='Sev' else (2 if row=='Maj2' else(3 if row=='Maj1' else(4 if row=='Mod' else(5 if row=='Min2' else(6 if row=='Min1' else 7)))))))
    trainInput.loc[:,'Functional']= trainInput.loc[:,'Functional'].map(lambda row: 0 if row=='Sal' else(1 if row=='Sev' else (2 if row=='Maj2' else(3 if row=='Maj1' else(4 if row=='Mod' else(5 if row=='Min2' else(6 if row=='Min1' else 7)))))))
    
    
    
    
    
    
    
    
    #neighberhood and year (do average price for every year) every 10 years and rank them  based on avg price per 10 year interval
    
    #year remodeled for every 20 years
    neigborhood=trainInput.loc[:,'Neighborhood']
    
    Blmngtn= neigborhood=="Blmngtn"
    temp1=trainDF.loc[Blmngtn]
    Blmngtnpr=temp1.loc[:,'SalePrice'].mean()
    
    Blueste=trainDF.loc[neigborhood=="Blueste"]
    Bluestepr=Blueste.loc[:,'SalePrice'].mean()
    
    BrDale=trainDF.loc[neigborhood=="BrDale"]
    BrDalepr=BrDale.loc[:,'SalePrice'].mean()
    
    BrkSide=trainDF.loc[neigborhood=="BrkSide"]
    BrkSidepr=BrkSide.loc[:,'SalePrice'].mean()
    
    ClearCr=trainDF.loc[neigborhood=="ClearCr"]
    ClearCrpr=ClearCr.loc[:,'SalePrice'].mean()
    
    CollgCr=trainDF.loc[neigborhood=="CollgCr"]
    CollgCrpr= CollgCr.loc[:,'SalePrice'].mean()
    
    Crawfor=trainDF.loc[neigborhood=="Crawfor"]
    Crawforpr= Crawfor.loc[:,'SalePrice'].mean()
    
    Edwards=trainDF.loc[neigborhood=="Edwards"]
    Edwardspr= Edwards.loc[:,'SalePrice'].mean()
    
    Gilbert=trainDF.loc[neigborhood=="Gilbert"]
    Gilbertpr=Gilbert.loc[:,'SalePrice'].mean()
    
    IDOTRR=trainDF.loc[neigborhood=="IDOTRR"]
    IDOTRRpr=IDOTRR.loc[:,'SalePrice'].mean()
    
    MeadowV=trainDF.loc[neigborhood=="MeadowV"]
    MeadowVpr= MeadowV.loc[:,'SalePrice'].mean()
    
    Mitchel=trainDF.loc[neigborhood=="Mitchel"]
    Mitchelpr=Mitchel.loc[:,'SalePrice'].mean()
    
    NAmes=trainDF.loc[neigborhood=="NAmes"]
    NAmespr=NAmes.loc[:,'SalePrice'].mean()
    
    NoRidge=trainDF.loc[neigborhood=="NoRidge"]
    NoRidgepr= NoRidge.loc[:,'SalePrice'].mean()
    
    NPkVill=trainDF.loc[neigborhood=="NPkVill"]
    NPkVillpr= NPkVill.loc[:,'SalePrice'].mean()
    
    NridgHt=trainDF.loc[neigborhood=="NridgHt"]
    NridgHtpr= NridgHt.loc[:,'SalePrice'].mean()
    
    NWAmes=trainDF.loc[neigborhood=="NWAmes"]
    NWAmespr= NWAmes.loc[:,'SalePrice'].mean()
    
    OldTown=trainDF.loc[neigborhood=="OldTown"]
    OldTownpr= OldTown.loc[:,'SalePrice'].mean()
    
    SWISU=trainDF.loc[neigborhood=="SWISU"]
    SWISUpr=SWISU.loc[:,'SalePrice'].mean()
    
    Sawyer=trainDF.loc[neigborhood=="Sawyer"]
    Sawyerpr=Sawyer.loc[:,'SalePrice'].mean()
    
    SawyerW=trainDF.loc[neigborhood=="SawyerW"]
    SawyerWpr= SawyerW.loc[:,'SalePrice'].mean()
    
    Somerst=trainDF.loc[neigborhood=="Somerst"]
    Somerstpr= Somerst.loc[:,'SalePrice'].mean()
    
    StoneBr=trainDF.loc[neigborhood=="StoneBr"]
    StoneBrpr = StoneBr.loc[:,'SalePrice'].mean()
    
    Timber=trainDF.loc[neigborhood=="Timber"]
    Timberpr= Timber.loc[:,'SalePrice'].mean()
    
    Veenker=trainDF.loc[neigborhood=="Veenker"]
    Veenkerpr= Veenker.loc[:,'SalePrice'].mean()
    
    trainInput.loc[:,'Neighborhood']=trainInput.loc[:,'Neighborhood'].map(lambda row:Blmngtnpr if row=="Blmngtn" else(Bluestepr if row=="Blueste" else(BrDalepr if row=="BrDale" else(BrkSidepr if row == "BrkSide" else(ClearCrpr if row == "ClearCr" else(CollgCrpr if row == "CollgCr" else(Crawforpr if row=="Crawfor" else(Edwardspr if row=="Edwards" else(Gilbertpr if row=="Gilbert" else(IDOTRRpr if row=="IDOTRR" else(MeadowVpr if row=="MeadowV" else(Mitchelpr if row=="Mitchel" else(NAmespr if row=="NAmes" else(NoRidgepr if row=="NoRidge" else(NPkVillpr if row=="NPkVill" else(NridgHtpr if row=="NridgHt" else(NWAmespr if row=="NWAmes" else(OldTownpr if row=="OldTown" else(SWISUpr if row=="SWISU" else(Sawyerpr if row=="Sawyer" else(SawyerWpr if row=="SawyerW" else(Somerstpr if row=="Somerst" else(StoneBrpr if row=="StoneBr" else(Timberpr if row=="Timber" else(Veenkerpr if row=="Veenker" else 0)))))))))))))))))))))))) )
    testInput.loc[:,'Neighborhood']=testInput.loc[:,'Neighborhood'].map(lambda row:Blmngtnpr if row=="Blmngtn" else(Bluestepr if row=="Blueste" else(BrDalepr if row=="BrDale" else(BrkSidepr if row == "BrkSide" else(ClearCrpr if row == "ClearCr" else(CollgCrpr if row == "CollgCr" else(Crawforpr if row=="Crawfor" else(Edwardspr if row=="Edwards" else(Gilbertpr if row=="Gilbert" else(IDOTRRpr if row=="IDOTRR" else(MeadowVpr if row=="MeadowV" else(Mitchelpr if row=="Mitchel" else(NAmespr if row=="NAmes" else(NoRidgepr if row=="NoRidge" else(NPkVillpr if row=="NPkVill" else(NridgHtpr if row=="NridgHt" else(NWAmespr if row=="NWAmes" else(OldTownpr if row=="OldTown" else(SWISUpr if row=="SWISU" else(Sawyerpr if row=="Sawyer" else(SawyerWpr if row=="SawyerW" else(Somerstpr if row=="Somerst" else(StoneBrpr if row=="StoneBr" else(Timberpr if row=="Timber" else(Veenkerpr if row=="Veenker" else 0)))))))))))))))))))))))) )
    
    
    
   
    
    #implement every 10 years 1872-2010
    #averaging the prices every 10 years for houses remodeled within the same 10 years
   
    yearblt=trainInput.loc[:,'YearRemodAdd']
    
    year=trainDF.loc[yearblt<1880]
    yearpr= year.loc[:,'SalePrice'].mean()
    trainInput.loc[:,'YearRemodAdd']=trainInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1880 else row)
    testInput.loc[:,'YearRemodAdd']=testInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1880 else row)

    year=trainDF.loc[yearblt<1900]
    yearpr= year.loc[:,'SalePrice'].mean()
    trainInput.loc[:,'YearRemodAdd']=trainInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1900 else row)
    testInput.loc[:,'YearBuilt']=testInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1900 else row)

    year=trainDF.loc[yearblt<1910]
    yearpr= year.loc[:,'SalePrice'].mean()
    trainInput.loc[:,'YearRemodAdd']=trainInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1910 else row)
    testInput.loc[:,'YearRemodAdd']=testInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1910 else row)

    year=trainDF.loc[yearblt<1920]
    yearpr= year.loc[:,'SalePrice'].mean()
    trainInput.loc[:,'YearRemodAdd']=trainInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1920 else row)
    testInput.loc[:,'YearRemodAdd']=testInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1920 else row)

    year=trainDF.loc[yearblt<1930]
    yearpr= year.loc[:,'SalePrice'].mean()
    trainInput.loc[:,'YearRemodAdd']=trainInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1930 else row)
    testInput.loc[:,'YearRemodAdd']=testInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1930 else row)

    year=trainDF.loc[yearblt<1940]
    yearpr= year.loc[:,'SalePrice'].mean()
    trainInput.loc[:,'YearRemodAdd']=trainInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1940 else row)
    testInput.loc[:,'YearRemodAdd']=testInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1940 else row)

    year=trainDF.loc[yearblt<1950]
    yearpr= year.loc[:,'SalePrice'].mean()
    trainInput.loc[:,'YearRemodAdd']=trainInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1950 else row)
    testInput.loc[:,'YearRemodAdd']=testInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1950 else row)

    year=trainDF.loc[yearblt<1960]
    yearpr= year.loc[:,'SalePrice'].mean()
    trainInput.loc[:,'YearRemodAdd']=trainInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1960 else row)
    testInput.loc[:,'YearRemodAdd']=testInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1960 else row)

    year=trainDF.loc[yearblt<1970]
    yearpr= year.loc[:,'SalePrice'].mean()
    trainInput.loc[:,'YearRemodAdd']=trainInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1970 else row)
    testInput.loc[:,'YearRemodAdd']=testInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1970 else row)

    year=trainDF.loc[yearblt<1980]
    yearpr= year.loc[:,'SalePrice'].mean()
    trainInput.loc[:,'YearRemodAdd']=trainInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1980 else row)
    testInput.loc[:,'YearRemodAdd']=testInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1980 else row)

    year=trainDF.loc[yearblt<1990]
    yearpr= year.loc[:,'SalePrice'].mean()
    trainInput.loc[:,'YearRemodAdd']=trainInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1990 else row)
    testInput.loc[:,'YearRemodAdd']=testInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<1990 else row)

    year=trainDF.loc[yearblt<2000]
    yearpr= year.loc[:,'SalePrice'].mean()
    trainInput.loc[:,'YearRemodAdd']=trainInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<2000 else row)
    testInput.loc[:,'YearRemodAdd']=testInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<2000 else row)

    year=trainDF.loc[yearblt<2010]
    yearpr= year.loc[:,'SalePrice'].mean()
    trainInput.loc[:,'YearRemodAdd']=trainInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<2010 else row)
    testInput.loc[:,'YearRemodAdd']=testInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<2010 else row)

    year=trainDF.loc[yearblt<2020]
    yearpr= year.loc[:,'SalePrice'].mean()
    trainInput.loc[:,'YearRemodAdd']=trainInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<2020 else row)
    testInput.loc[:,'YearRemodAdd']=testInput.loc[:,'YearRemodAdd'].map(lambda row: yearpr if row<2020 else row)

    
    
    #Averaging the praces of years built or remodeled within the same 20 years
    
    trainDF.loc[:,'YearBuilt']=trainDF.apply(lambda row: row['YearBuilt'] if row['YearBuilt']>row['YearRemodAdd'] else row['YearRemodAdd'], axis=1)
    
    testDF.loc[:,'YearBuilt']=testDF.apply(lambda row: row['YearBuilt'] if row['YearBuilt']>row['YearRemodAdd'] else row['YearRemodAdd'], axis=1)
    
    trainInput.loc[:,'YearBuilt']=trainDF.apply(lambda row: row['YearBuilt'] if row['YearBuilt']>row['YearRemodAdd'] else row['YearRemodAdd'], axis=1)
    
    testInput.loc[:,'YearBuilt']=testDF.apply(lambda row: row['YearBuilt'] if row['YearBuilt']>row['YearRemodAdd'] else row['YearRemodAdd'], axis=1)
    
    yearblt=trainDF.loc[:,'YearBuilt']
    
    
    
    year=trainDF.loc[yearblt<1880]
    
    yearpr=year.loc[:,'SalePrice'].mean() 
    
    trainInput.loc[:,'YearBuilt']=trainInput.loc[:,'YearBuilt'].map(lambda row: yearpr if row<1880 else row)
    testInput.loc[:,'YearBuilt']=testInput.loc[:,'YearBuilt'].map(lambda row: yearpr if row<1880 else row)

    
    
    year=trainDF.loc[yearblt<1900]
    
    yearpr=year.loc[:,'SalePrice'].mean() 
    
    trainInput.loc[:,'YearBuilt']=trainInput.loc[:,'YearBuilt'].map(lambda row: yearpr if row<1900 else row)
    testInput.loc[:,'YearBuilt']=testInput.loc[:,'YearBuilt'].map(lambda row: yearpr if row<1900 else row)

    
    
    year=trainDF.loc[yearblt<1920]
    
    yearpr=year.loc[:,'SalePrice'].mean() 
    
    trainInput.loc[:,'YearBuilt']=trainInput.loc[:,'YearBuilt'].map(lambda row: yearpr if row<1920 else row)
    testInput.loc[:,'YearBuilt']=testInput.loc[:,'YearBuilt'].map(lambda row: yearpr if row<1920 else row)

   
    
    year=trainDF.loc[yearblt<1940]
    
    yearpr=year.loc[:,'SalePrice'].mean() 
    
    trainInput.loc[:,'YearBuilt']=trainInput.loc[:,'YearBuilt'].map(lambda row: yearpr if row<1940 else row)
    testInput.loc[:,'YearBuilt']=testInput.loc[:,'YearBuilt'].map(lambda row: yearpr if row<1940 else row)

    
    year=trainDF.loc[yearblt<1960]
    
    yearpr=year.loc[:,'SalePrice'].mean() 
    
    trainInput.loc[:,'YearBuilt']=trainInput.loc[:,'YearBuilt'].map(lambda row: yearpr if row<1960 else row)
    testInput.loc[:,'YearBuilt']=testInput.loc[:,'YearBuilt'].map(lambda row: yearpr if row<1960 else row)

    
    year=trainDF.loc[yearblt<1980]
    
    yearpr=year.loc[:,'SalePrice'].mean() 
    
    trainInput.loc[:,'YearBuilt']=trainInput.loc[:,'YearBuilt'].map(lambda row: yearpr if row<1980 else row)
    testInput.loc[:,'YearBuilt']=testInput.loc[:,'YearBuilt'].map(lambda row: yearpr if row<1980 else row)
    
    year=trainDF.loc[yearblt<2000]
    
    yearpr=year.loc[:,'SalePrice'].mean() 
    
    trainInput.loc[:,'YearBuilt']=trainInput.loc[:,'YearBuilt'].map(lambda row: yearpr if row<2000 else row)
    testInput.loc[:,'YearBuilt']=testInput.loc[:,'YearBuilt'].map(lambda row: yearpr if row<2000 else row)

    
    year=trainDF.loc[yearblt<2020]
    
    yearpr=year.loc[:,'SalePrice'].mean() 
    
    trainInput.loc[:,'YearBuilt']=trainInput.loc[:,'YearBuilt'].map(lambda row: yearpr if row<2020 else row)
    testInput.loc[:,'YearBuilt']=testInput.loc[:,'YearBuilt'].map(lambda row: yearpr if row<2020 else row)

    
 
   
    
   
      
    #Standardize all the large data
    inputcols=['PoolArea','MiscVal','GrLivArea','TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','LotArea','GrLivArea','MasVnrArea',"GarageArea","LotFrontage",'1stFlrSF', '2ndFlrSF']
    standardize(trainInput,inputcols)
    standardize(testInput,inputcols)
    
    
    
    
    #normalize all the small data
    #'TotRmsAbvGrd','GarageCond','Fence','GarageQual','BsmtExposure','HeatingQC','Utilities','GrLivArea','KitchenAbvGr','BedroomAbvGr','TotalBsmtSF',"CentralAir",'Alley',"Functional",'1stFlrSF', '2ndFlrSF', "ExterQual","BsmtCond","ExterCond","TotRmsAbvGrd","GarageArea","OverallQual","OverallCond",'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath'
    inputcols1=['YearBuilt','YearRemodAdd','Neighborhood','Electrical','KitchenQual','PavedDrive','PoolQC','TotRmsAbvGrd','GarageCars','GarageCond','Fence','GarageQual','BsmtExposure','HeatingQC','Utilities','KitchenAbvGr','BedroomAbvGr',"Functional",'Alley',"CentralAir",'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',"ExterQual","BsmtCond","ExterCond","TotRmsAbvGrd","OverallQual","OverallCond"]
    normalize(trainInput,inputcols1)
    normalize(testInput,inputcols1)
    
     #Final test that there is no NaN in our data to fit our model
    #'TotRmsAbvGrd','GarageCars','GrLivArea','KitchenAbvGr','BedroomAbvGr','TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','GrLivArea',"Functional",'Alley',"CentralAir","GarageArea", '2ndFlrSF',"ExterQual","BsmtCond","ExterCond","TotRmsAbvGrd","OverallQual","OverallCond"
    inputcollls=['YearBuilt','YearRemodAdd','Neighborhood','Electrical','KitchenQual','PavedDrive','PoolQC','PoolArea','MiscVal','TotRmsAbvGrd','GarageCars','GarageCond','Fence','GarageQual','BsmtExposure','HeatingQC','Utilities','GrLivArea','KitchenAbvGr','BedroomAbvGr','TotalBsmtSF','BsmtFinSF1','BsmtFinSF2','LotArea','GrLivArea','MasVnrArea',"Functional",'Alley',"CentralAir",'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',"GarageArea","LotFrontage",'1stFlrSF', '2ndFlrSF',"ExterQual","BsmtCond","ExterCond","TotRmsAbvGrd","OverallQual","OverallCond"]
    replacnanw0(trainInput,inputcollls)
    replacnanw0(testInput,inputcollls)  
    
    #print(trainInput.loc[:,'Neighborhood'])

    '''
    Any transformations you do on the trainInput will need to be done on the
    testInput the same way. (For example, using the exact same min and max, if
    you're doing normalization.)
    '''
    
    trainOutput = trainDF.loc[:, 'SalePrice']
    testIDs = testDF.loc[:, 'Id']
    
    #trainInput = trainInput.loc[:, predictor]
    
    return trainInput, testInput, trainOutput, testIDs, predictors
    
#helper functions for pre-processing
def replacnanw0(DataFrame,atribNames):
    DataFrame.loc[:, atribNames] = DataFrame.loc[:, atribNames].fillna(0)

def replacewmode(DataFrame,atribNames):
    DataFrame.loc[:, atribNames] = DataFrame.loc[:, atribNames].fillna(DataFrame.loc[:, atribNames].mode().iloc[0])
    
def replacewmean(DataFrame,atribNames):
    DataFrame.loc[:, atribNames] = DataFrame.loc[:, atribNames].fillna(DataFrame.loc[:, atribNames].mean())
    
def normalize(DataFrame, someCols):
    DataFrame.loc[:,someCols]=DataFrame.apply(lambda row: (row-DataFrame.loc[:,someCols].min())/(DataFrame.loc[:,someCols].max()-DataFrame.loc[:,someCols].min()), axis=1 )
        
def standardize(DataFrame, atribNames):
    DataFrame.loc[:,atribNames]=DataFrame.apply(lambda row: (row-DataFrame.loc[:,atribNames].mean())/DataFrame.loc[:,atribNames].std(), axis=1 )
    
# ===============================================================================
'''
Demonstrates some provided helper functions 
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')

# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) 
    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================

if __name__ == "__main__":
    main()

