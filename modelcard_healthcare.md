
# Model Card


<table style="width: 100%;">
  <tr style="background-color:#EDF0FF">
    <td style="width: 50%;"><strong>Name:</strong>  <<modelop.useCases[0].modelMetaData.name>> <br><strong>Developer:</strong> <<modelop.useCases[0].modelMetaData.custom.accountability.(31111).(answer)>> </td>
    <td style="width: 50%;"><strong>Inquiries or to report an issue (Business Owner): </strong><<modelop.useCases[0].modelMetaData.custom.accountability.(31).(answer)>><br></td>
  </tr>
  <tr>
    <td><strong>Release Stage:</strong> <<modelop.useCases[0].modelMetaData.modelStage>> <br> <strong>Global Availability:</strong> <<modelop.useCases[0].modelMetaData.custom.overview.(14).(answer)>> </td>
    <td><strong>Version: </strong> <<modelop.deployableModel.metaData.name>></td>
  </tr>
  <tr style="background-color:#EDF0FF">
    <td><strong>Summary:</strong> <<modelop.useCases[0].modelMetaData.description>> <br><br><strong>Keywords:</strong> <<modelop.useCases[0].modelMetaData.modelUseCategory>> </td>
    <td><strong>Uses and Directions:</strong><br> * <strong>Intended use and workflow:</strong> <<modelop.useCases[0].modelMetaData.custom.scope_usages_limitations.(3).(answer)>> <br> * <strong>Primary intended users: </strong> <<modelop.useCases[0].modelMetaData.custom.scope_usages_limitations.(1).(answer)>> <br> * <strong>If clinical, is an HCP responsible for AI deision?: </strong> <<modelop.useCases[0].modelMetaData.custom.solution_details.(3).(answer)>>  <br> * <strong>Targeted patient population: </strong> <<modelop.useCases[0].modelMetaData.custom.(scope_usages_limitations).(4).(answer)>> <br> * <strong>Cautioned out-of-scope settings and use cases:</strong> <<modelop.useCases[0].modelMetaData.custom.(scope_usages_limitations).(5).(answer)>> </td>  
  </tr>
</table>

<table style="width: 100%; margin-top: 20px;">
  <tr>
    <td style="width: 100%;"><strong>Warnings:</strong><br> * <strong>Known risks and limitations: </strong> <<modelop.useCases[0].modelMetaData.custom.(scope_usages_limitations).(6).(answer)>><br> * <strong>Known biases or ethical considerations: </strong> <<modelop.useCases[0].modelMetaData.custom.(scope_usages_limitations).(7).(answer)>> <br> * <strong>Clinical risk level: </strong> <<modelop.useCases[0].modelMetaData.custom.riskRating.clinicalRiskFinal>> </td>
  </tr>
  <tr style="background-color:#EDF0FF">
    <td><strong>Trust Ingredients:</strong><br><strong>AI System Facts:</strong><br> * <strong>Technology Overview and outputs:</strong> <<modelop.useCases[0].modelMetaData.custom.(implementation_and_policy_adhe).(1).(answer)>><br> * <strong>Technology is internal-build or vendor solution? </strong> <<modelop.useCases[0].modelMetaData.custom.solution_details.(1).(answer)>><br> * <strong>Model type: </strong><<modelop.storedModel.modelMetaData.modelMethodology>> <br> * <strong>Foundation models used?:</strong> <<modelop.useCases[0].modelMetaData.custom.(implementation_and_policy_adhe).(2).(answer)>> <br> * <strong>Data Sensitivity Classification: </strong><<modelop.useCases[0].modelMetaData.custom.data.(1).(answer)>> <br> * <strong>Solution uses Patient Data?: </strong><<modelop.useCases[0].modelMetaData.custom.data.(2).(answer)>> <br> * <strong>Patient data available to Vendor (if applicable)?: </strong><<modelop.useCases[0].modelMetaData.custom.data.(3).(answer)>> <br> * <strong>Solution needs to be adapted/trained for the Organization?: </strong><<modelop.useCases[0].modelMetaData.custom.data.(14).(answer)>> <br> * <strong>Vendor has IP rights to trained model (if applicable)?: </strong><<modelop.useCases[0].modelMetaData.custom.data.(15).(answer)>> <br> * <strong>If vendor needs to re-train with Organization data, which data (if applicable)?: </strong><<modelop.useCases[0].modelMetaData.custom.data.(16).(answer)>><br><br> <strong>Ongoing Maintenance:</strong><br> * <strong>After implementation, how will the performance of the AI solution be assessed?:</strong> </strong><<modelop.useCases[0].modelMetaData.custom.(solution_details).(3).(answer)>> <br> * <strong>Is the technology able to report adverse events?:</strong> </strong><<modelop.useCases[0].modelMetaData.custom.(implementation_and_policy_adhe).(5).(answer)>>  </td>
  </tr>
  <tr>
    <td><strong>Transparency, Intelligibility, and Accountability mechanisms, if applicable:</strong><br> * <strong>Accountable AI Governance Officer:</strong><<modelop.useCases[0].modelMetaData.custom.accountability.(311111).(answer)>><br> * <strong>Accountable Legal Officer: </strong><<modelop.useCases[0].modelMetaData.custom.accountability.(3111111).(answer)>><br> * <strong>Independent Reviewer (if applicable):</strong><<modelop.useCases[0].modelMetaData.custom.accountability.(31111111).(answer)>><br> * <strong>Accountable Legal Officer:</strong><<modelop.useCases[0].modelMetaData.custom.accountability.(3111111).(answer)>></td>
  </tr>
  <tr style="background-color:#EDF0FF">
    <td><strong>Resources:</strong><br> * <strong>Funding source of the technical implementation:</strong> <<modelop.useCases[0].modelMetaData.custom.overview.(3).(answer)>><br> * <strong>Clinical Trial, if Available:</strong> <<modelop.useCases[0].modelMetaData.custom.(implementation_and_policy_adhe).(25).(answer)>> <br> * <strong>Peer Reviewed Publication(s):</strong> <<modelop.useCases[0].modelMetaData.custom.(implementation_and_policy_adhe).(26).(answer)>> <br> * <strong>Patient consent or disclosure required or suggested:</strong> <<modelop.useCases[0].modelMetaData.custom.overview.(21).(answer)>> <br> <strong>* Model Documentation: </strong><a href="<<modelop.storedModel.modelAssets.[assetRole=MODEL_DOCUMENTATION].fileUrl>>"><<modelop.storedModel.modelAssets.[assetRole=MODEL_DOCUMENTATION].filename>></a></td>
  </tr>
</table>


## Baseline Data

- **Data Set Name:** <<modelop.storedModel.modelAssets.[assetRole=BASELINE_DATA].filename>>

- **Repository:** <<modelop.storedModel.modelAssets.[assetRole=BASELINE_DATA].fileUrl>>


## Model Test Results: <br>
**Performance:**

|<<modelop.modelTestResult.testResults.(performance)[0].values>>|

**Stability:**

<<modelopgraph.stability.*>>

**Ethical Fairness / Bias:**

<<modelopgraph.groupbias.gender>>


**NOTE:** For instructions, references, resources, contributors, and disclaimers, please refer to the full documentation located at [www.chai.org](http://www.chai.org).

This document is licensed under a **Creative Commons Attribution-Non-Commercial-No Derivatives 4.0 International License (CC BY-NC-ND 4.0).**

You are free to share this material (copy and redistribute it in any medium or format) under the following terms:

- **Attribution:** <<modelop.useCases[0].modelMetaData.name>>
- **Noncommercial:** You may not use the material for commercial purposes.
- **No Derivatives:** If you remix, transform, or build upon the material, you may not distribute the modified material.

For more information about this license, visit [creativecommons.org/licenses/by-nc-nd/4.0/](https://creativecommons.org/licenses/by-nc-nd/4.0/).


