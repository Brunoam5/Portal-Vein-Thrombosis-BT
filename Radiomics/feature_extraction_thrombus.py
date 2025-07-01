import os
import pandas as pd
from radiomics import featureextractor 

out_path_slices = 'Introduce path to CT'
out_path_thrombus = 'Introduce path to Thrombus mask'

params = 'Introduce path to params.yaml'
extractor = featureextractor.RadiomicsFeatureExtractor(params)

def get_valid_nii_files(directory):
    return [f for f in sorted(os.listdir(directory)) if f.endswith('.nii')]

dft = []
for element in get_valid_nii_files(out_path_slices):
    imageName = os.path.join(out_path_slices, element)
    maskName = os.path.join(out_path_thrombus, element)

    if os.path.exists(maskName):
        try:
            result = extractor.execute(imageName, maskName)
            df = pd.DataFrame.from_dict(result, orient='index')
            dft.append(df)
        except Exception as e:
            print(f"Error processing {element}: {e}")
    else:
        print(f"Mask not found for {element}")

dft_thrombus = pd.concat(dft, axis=1).T
dft_thrombus = dft_thrombus.reset_index().iloc[:, 38:]
print(dft_thrombus.columns.tolist())

dft_thrombus.to_csv('Introduce output path for csv file', index=False)
del df, dft, element, extractor, imageName, maskName, out_path_thrombus, out_path_slices, params, result
