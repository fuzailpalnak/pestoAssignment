# pestoAssignment

## Installation

```sh
pip install -r requirements.txt
```

## Preprocessing Techniques Summary

### Text Normalization

- **Technique**: Text normalization involves removing punctuation, expanding contractions, and converting text to lowercase.
- **Advantages**:
  - **Improved Consistency**: Ensures uniform language usage throughout the dataset.
  - **Simplifies Processing**: Clean text facilitates easier subsequent NLP tasks.

### Paraphrasing

- **Technique**: Utilizes the TextGenie model (`"ramsrigouthamg/t5_paraphraser"`) to generate paraphrases.
- **Advantages**:
  - **Data Augmentation**: Increases dataset size with varied examples.
  - **Enhanced Diversity**: Provides alternative phrasing while retaining original meaning.
  - **Contextual Adaptability**: Can adapt paraphrases based on context.



## Train
```python
from pesto.run import run_from_raw_data
run_from_raw_data()
```

## Inference

Download Weights from "[here](https://github.com/fuzailpalnak/pestoAssignment/releases/download/v0.0.1/saved_model.zip)" and follow the notebook - https://github.com/fuzailpalnak/pestoAssignment/blob/main/pestoAssignmentInference.ipynb
