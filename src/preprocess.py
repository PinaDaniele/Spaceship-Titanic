import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(df):
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP']

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df['Cabin'] = df['Cabin'].fillna('U/0/U')

    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
    group_size_dict = df['Group'].value_counts().to_dict()
    df['GroupSize'] = df['Group'].map(group_size_dict)

    df['TotalSpend'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']

    df = df.drop(columns=['PassengerId', 'Cabin', 'Name', 'Group'])

    df['Num'] = df['Num'].astype(int)
    df['CryoSleep'] = df['CryoSleep'].astype(int)
    df['VIP'] = df['VIP'].astype(int)

    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination', 'Deck', 'Side']).astype(int)

    scaler = StandardScaler()
    scale_cols = num_cols + ['Num', 'GroupSize', 'TotalSpend']
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    return df

if __name__ == '__main__':
    df = pd.read_csv('../data/train.csv')
    print(clean_data(df).head())
