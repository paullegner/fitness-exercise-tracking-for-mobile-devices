import pandas as pd

def class_count(df):
    print(df['class'].value_counts())

def handle_blank_inputs(df):
    print('Before:')
    class_count(df)
    with pd.option_context('mode.use_inf_as_na', True):
        df.dropna(inplace=True)
    print('After:')
    class_count(df)

if __name__ == '__main__':
    data = pd.read_csv('/Users/paul1/Desktop/test_keypoints.csv')
    handle_blank_inputs(data)

