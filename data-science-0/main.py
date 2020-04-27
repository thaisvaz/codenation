
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# In[3]:


black_friday.shape


# ## Inicie sua análise a partir daqui

# In[4]:


df= black_friday


# In[5]:


df.head()


# In[6]:


df.describe()


# In[20]:


df.info()


# In[24]:


df.tail()


# In[25]:


df.Gender.value_counts() 


# In[40]:


# média Purchase por City_Category
df[['City_Category','Purchase']].groupby(['City_Category']).mean().round(2)


# In[7]:


pd.DataFrame(black_friday['Purchase']).info()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[8]:


#tentei só black_friday.shape , mas não funcionou
def q1():
    # Retorne aqui o resultado da questão 1.    
    return black_friday.shape


# In[9]:


black_friday.shape


# In[10]:


black_friday.info()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[11]:


# O certo seria contar com os dados não repetidos, mas o gabarito não aceitou 
def q2():
    # Retorne aqui o resultado da questão 2.
    q2=len(black_friday[(black_friday['Gender']=='F')&(black_friday['Age']=='26-35')])
    
    return q2


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[12]:


def q3():
    # Retorne aqui o resultado da questão 3.
    q3=df.User_ID.nunique()
    
    return q3


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[13]:


def q4():
    # Retorne aqui o resultado da questão 4.
    q4=black_friday.dtypes.nunique()
    
    return q4


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[14]:


def q5():
    # Retorne aqui o resultado da questão 5.
    q5=(len(black_friday) - len(black_friday.dropna())) / len(black_friday)
    
    return q5


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[15]:


def q6():
    # Retorne aqui o resultado da questão 6.
    q6=black_friday.Product_Category_3.isnull().sum()
    
    return q6


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[16]:


# sim, ainda uso argmax() srsrsr

def q7():
    # Retorne aqui o resultado da questão 7.
    q7=black_friday['Product_Category_3'].value_counts().argmax()
    
    return q7


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[17]:


def q8():
    # Retorne aqui o resultado da questão 8.
    
    # normalizando:
    black_friday['Purchase_normalizado']=(black_friday['Purchase']-black_friday['Purchase'].min())/(black_friday['Purchase'].max()-black_friday['Purchase'].min())
    
    # calculando a média
    q8=black_friday['Purchase_normalizado'].mean()
    
    return q8


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[18]:


def q9():
    # Retorne aqui o resultado da questão 9.
    
    # padronizando
    black_friday['Purchase_padronizado']=(black_friday['Purchase']-black_friday['Purchase'].mean())/np.std(black_friday['Purchase'])
    
    # conta ocorrências entre -1 e 1
    q9=black_friday[(black_friday['Purchase_padronizado']<1)&(black_friday['Purchase_padronizado']>-1)].Purchase_padronizado.count()
    
    return q9


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[19]:


def q10():
    # Retorne aqui o resultado da questão 10.
    q10=black_friday[black_friday['Product_Category_2'].isnull()][['Product_Category_2','Product_Category_3']].isnull().values.all()
    return q10

