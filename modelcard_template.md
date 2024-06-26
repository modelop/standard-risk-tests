# Model Card for <<modelop.storedModel.modelMetaData.name>>

<!-- Provide a quick summary of what the model is/does. -->
<<modelop.storedModel.modelMetaData.description>>

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

- **Model Use Case:** <<modelop.deployableModel.associatedModels.[associationRole=MODEL_USE_CASE].associatedModel.storedModel.modelMetaData.name>>

- **Developed by:** <<modelop.storedModel.createdBy>>
- **Model type:** <<modelop.storedModel.modelMetaData.modelMethodology>> - <<modelop.storedModel.modelMetaData.type>>
- **Model Documentation:** <a href="<<modelop.storedModel.modelAssets.[assetRole=MODEL_DOCUMENTATION].fileUrl>>"><<modelop.storedModel.modelAssets.[assetRole=MODEL_DOCUMENTATION].filename>></a>

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** <<modelop.storedModel.modelMetaData.repositoryInfo.repositoryRemote>> **branch:** <<modelop.storedModel.modelMetaData.repositoryInfo.repositoryBranch>>

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

[More Information Needed]

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

```
<<modelop.storedModel.modelAssets.[assetRole=MODEL_TEST_SOURCE].sourceCode>>
```
## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->
### Dataset Card for <<modelop.storedModel.modelAssets.[assetRole=TRAINING_DATA].filename>>

#### Dataset Sources

- **Repository:** <<modelop.storedModel.modelAssets.[assetRole=TRAINING_DATA].fileUrl>>

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

- **Training regime:** [More Information Needed] <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->
### Dataset Card for <<modelop.storedModel.modelAssets.[assetRole=BASELINE_DATA].filename>>

#### Dataset Sources

- **Repository:** <<modelop.storedModel.modelAssets.[assetRole=BASELINE_DATA].fileUrl>>

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

|Category| Passes                                                                                                       | Reason                                                                                                       |
|-|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
|Performance| <<modelop.modelTestResult.dmnRuleResults.[testCategory=Performance].passes>>                                 | <<modelop.modelTestResult.dmnRuleResults.[testCategory=Performance].reason>>                                 |
|Data Drift - Kolmogorov Smirnov| <<modelop.modelTestResult.dmnRuleResults.[testCategory=KS Data Drift].passes>>                               | <<modelop.modelTestResult.dmnRuleResults.[testCategory=KS Data Drift].reason>>                               |
|Concept Drift - Kolmogorov Smirnov| <<modelop.modelTestResult.dmnRuleResults.[testCategory=KS Concept Drift].passes>>                            | <<modelop.modelTestResult.dmnRuleResults.[testCategory=KS Concept Drift].reason>>                            |
|Characteristic Stability| <<modelop.modelTestResult.dmnRuleResults.[testCategory=Characteristic Stability].passes>>                    | <<modelop.modelTestResult.dmnRuleResults.[testCategory=Characteristic Stability].reason>>                    |
|Bias Disparity| <<modelop.modelTestResult.dmnRuleResults.[testCategory=Bias Disparity].passes>>                              | <<modelop.modelTestResult.dmnRuleResults.[testCategory=Bias Disparity].reason>>                              |
|Autocorrelation| <<modelop.modelTestResult.dmnRuleResults.[testCategory=Autocorrelation].passes>>                             | <<modelop.modelTestResult.dmnRuleResults.[testCategory=Autocorrelation].reason>>                             |
|Homoscedacticity: Breusch Pagan| <<modelop.modelTestResult.dmnRuleResults.[testCategory=Homoscedacticity: Breusch Pagan].passes>>             | <<modelop.modelTestResult.dmnRuleResults.[testCategory=Homoscedacticity: Breusch Pagan].reason>>             |
|Homoscedacticity: Engle Lagrange Multiplier| <<modelop.modelTestResult.dmnRuleResults.[testCategory=Homoscedacticity: Engle Lagrange Multiplier].passes>> | <<modelop.modelTestResult.dmnRuleResults.[testCategory=Homoscedacticity: Engle Lagrange Multiplier].reason>> |
|Homoscedacticity: Ljung Box Q| <<modelop.modelTestResult.dmnRuleResults.[testCategory=Homoscedacticity: Ljung Box Q].passes>>               | <<modelop.modelTestResult.dmnRuleResults.[testCategory=Homoscedacticity: Ljung Box Q].reason>>               |
|Normality: Kolmogorov Smirnov| <<modelop.modelTestResult.dmnRuleResults.[testCategory=Normality: Kolmogorov Smirnov].passes>>               | <<modelop.modelTestResult.dmnRuleResults.[testCategory=Normality: Kolmogorov Smirnov].reason>>               |
|Normality: Anderson Darling| <<modelop.modelTestResult.dmnRuleResults.[testCategory=Normality: Anderson Darling].passes>>                 | <<modelop.modelTestResult.dmnRuleResults.[testCategory=Normality: Anderson Darling].reason>>                 |
|Normality: Cramer Von Mises| <<modelop.modelTestResult.dmnRuleResults.[testCategory=Normality: Cramer Von Mises].passes>>                 | <<modelop.modelTestResult.dmnRuleResults.[testCategory=Normality: Cramer Von Mises].reason>>                 |
|Linearity: Pearson Correlation| <<modelop.modelTestResult.dmnRuleResults.[testCategory=Linearity: Pearson Correlation].passes>>              | <<modelop.modelTestResult.dmnRuleResults.[testCategory=Linearity: Pearson Correlation].reason>>              |
|Mulitcolinearity| <<modelop.modelTestResult.dmnRuleResults.[testCategory=Mulitcolinearity].passes>>                            | <<modelop.modelTestResult.dmnRuleResults.[testCategory=Mulitcolinearity].reason>>                            |

- **Performance Metrics:**<br>

|<<modelop.modelTestResult.testResults.(performance)[0].values>>|

- **Stability Analysis:**<br> <<modelopgraph.stability.*>>

### Results

[More Information Needed]

#### Summary

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Model Card Contact

<<modelop.storedModel.createdBy>>


