import streamlit as st
import pandas as pd
import copy
import scipy.sparse as sparse
import implicit

train_ratio = 0.7
factors = 20
regularization = 0.1
iterations = 20
alpha = 15
N = 10**3

@st.cache
def get_data():
    product_dic = pd.read_csv('data/products.csv')
    product_dic = product_dic[['product_id','product_name']]

    df_users = pd.read_csv('data/orders.csv')
    df_users = df_users[df_users['eval_set'] == 'prior']
    df_users = df_users[['order_id','user_id']]

    df_products = pd.read_csv('data/order_products__prior.csv')
    df_products = df_products[['order_id','product_id']]

    df_merged = pd.merge(df_users, df_products, on='order_id', how='left')
    df_merged = df_merged[['user_id','product_id']]
    df_merged['count'] = 1
    df_merged = df_merged.groupby(['user_id','product_id'], as_index=False)['count'].sum()

    df_train = df_merged.sample(frac=train_ratio)
    df_test = (df_merged.merge(df_train, on=['user_id','product_id'], how='left', indicator=True)
               .query('_merge == "left_only"').drop('_merge', 1))
    df_test = df_test.drop(['count_y'],axis=1)
    df_test.columns = ['user_id', 'product_id', 'count_purchase']

    return product_dic, df_train, df_merged

@st.cache
def build_model(factors, regularization, iterations, alpha):
    _,df_train,_ = get_data()
    sparse_item_user = sparse.csr_matrix(
        (df_train['count'].astype(float), (df_train['product_id'], df_train['user_id'])))
    sparse_user_item = sparse.csr_matrix(
        (df_train['count'].astype(float), (df_train['user_id'], df_train['product_id'])))
    data_conf = (sparse_item_user * alpha).astype('double')

    model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations)
    model.fit(data_conf)

    return model, sparse_user_item

def output_recs(recommended, named):
    recs = [item[0] for item in recommended]
    if named:
        product_dic,_,_ = get_data()
        product_names = []
        for product_id in product_ids:
            product_names.append(product_dic.product_name.loc[product_dic.product_id == product_id].iloc[0])
        return pd.DataFrame({'product': product_names})
    else:
        return recs

def similar_products(product_id, N, named=False):
    inst = copy.deepcopy(model)
    similar = inst.similar_items(product_id, N + 1)
    return output_recs(similar[1:],named)

def recommendations(user_id, N, named=False):
    inst = copy.deepcopy(model)
    recommended = inst.recommend(user_id, sparse_user_item, N=N)
    return output_recs(recommended,named)

st.title('Recommendation Engine')
st.subheader('Made by Div Dasani')
st.write('\n')
st.write('\n')
st.sidebar.header('Search')

product_dic, _, df_merged = get_data()
model, sparse_user_item = build_model(factors, regularization, iterations, alpha)

user_id = st.sidebar.text_input('User ID', '202279')
if st.sidebar.button('User Recommendations'):
    try:
        user_id = int(user_id)
        if user_id > 206209 or user_id < 1:
            st.error('Please input an integer between 1 and 206209')
        else:
            st.write('User Purchase History')
            st.write(name(df_merged[df_merged['user_id'] == user_id].product_id.tolist()))
            st.write('\nRecommended Products')
            st.write(recommendations(user_id, 5, named=True))
    except ValueError:
        st.error('Please input an integer between 1 and 206209')

item_name = st.sidebar.selectbox('Item Name',product_dic.product_name.unique()[:100])
if st.sidebar.button('Similar Items'):
    target_id = product_dic.product_id.loc[product_dic.product_name == item_name].iloc[0]
    st.write('Similar Products to {}:'.format(item_name))
    st.write(similar_products(target_id,5,named=True))
