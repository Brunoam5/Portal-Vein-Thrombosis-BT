# Automated Detection, Segmentation, and Radiomics-Based Analysis for Prognostic Prediction of Portal Vein Thrombosis

Radiomics has emerged as a promising tool for non-invasive prediction of treatment outcomes in several pathologies, included vascular disorders. This study aimed to develop and validate a radiomics-based pipeline for the prediction of thrombus recanalization following anticoagulant therapy in patients with portal vein thrombosis (PVT). The pipeline integrates automatic thrombus segmentation with the extraction and selection of radiomic features from contrast-enhanced (CE) computed tomography (CT) images.
A retrospective dataset of 46 PVT patients was collected, manually segmented, and pre-processed. Two deep learning (DL) models—nnUNet and nnSAM—were evaluated for automatic thrombus segmentation using Dice and Surface Dice metrics. Radiomic features (first-order, texture, and higher-order) were extracted using PyRadiomics and reduced through statistical filtering (variance thresholding, Spearman correlation, and Mann-Whitney U test). Feature stability was assessed via bootstrapping, and classification was performed using Elastic Net, Random Forest (RF), Extreme Gradient Boosting (XGBoost), and Support Vector Machine (SVM) Linear models.
Seven radiomic features were identified as stable and predictive across classifiers. The highest performance was achieved using radiomic features alone, and no negative impact was observed when integrating manual variables such as vessel occlusion. Instead, different types of features (manual and radiomic) complemented each other. The variable “Occlusion_RightPV” was consistently selected across models and reached statistical significance.
These findings demonstrate the feasibility of combining DL-based segmentation with radiomics and machine learning (ML) for outcome prediction in PVT and highlight the value of integrating anatomical and imaging biomarkers for personalized disease management.


<img width="1194" alt="Captura de pantalla 2025-06-05 a las 11 12 47" src="https://github.com/user-attachments/assets/f08ff288-73b2-43ff-86a0-0e038f0fc68d" />






