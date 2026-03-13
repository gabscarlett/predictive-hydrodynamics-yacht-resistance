import pandas as pd


def load_data(filepath):
    col_names = ['Centre_buoyancy', 'Prismatic_coeff', 'Len_displacement_ratio', 'Beam_draught_ratio', 
                 'Len_beam_ratio', 'Froude_number', 'Resistance']
    df = pd.read_csv(filepath)
    df.columns = col_names
    X = df.drop(columns=['Resistance']) # features
    y = df['Resistance'] # target
    return X, y
