#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[10]:


startup = pd.read_csv("50_Startups1.csv")


# In[11]:


startup.head(40)


# In[12]:


startup.corr()


# In[13]:


import seaborn as sns
sns.pairplot(startup)


# In[14]:


startup.columns


# In[15]:


import statsmodels.formula.api as smf


# In[16]:


ml1 = smf.ols('ms~Pr+rd+ad',data=startup).fit()


# In[17]:


ml1.params


# In[18]:


ml1.summary()


# In[19]:


ml_v=smf.ols('ms~rd',data = startup).fit() 


# In[20]:


ml_v.summary()


# In[21]:


ml_w=smf.ols('ms~ad',data = startup).fit()


# In[22]:


ml_w.summary()


# In[25]:


ml_wv=smf.ols('ms~Pr+rd',data = startup).fit()


# In[26]:


ml_wv.summary()


# In[27]:


import statsmodels.api as sm
sm.graphics.influence_plot(ml1)


# In[30]:


startup_new=startup.drop(startup.index[[47,48]],axis=0)
startup_new


# In[31]:


ml1_new = smf.ols('ms~Pr+rd+ad',data=startup).fit()


# In[32]:


ml1_new.params


# In[34]:


ml1_new.summary()


# In[35]:


startup_pred = ml1_new.predict(startup_new[['ms','Pr','rd','ad']])
startup_pred


# In[36]:


sm.graphics.plot_partregress_grid(ml1_new)


# In[38]:


final_ml= smf.ols('ms~Pr+rd+ad',data = startup_new).fit()
final_ml.params
final_ml.summary()


# In[39]:


sm.graphics.plot_partregress_grid(final_ml)


# In[41]:


plt.scatter(startup_new.Pr,startup_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")


# In[42]:


plt.scatter(startup_pred,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


# In[43]:


plt.hist(final_ml.resid_pearson) 


# In[44]:


import pylab          
import scipy.stats as st


# In[45]:


st.probplot(final_ml.resid_pearson, dist="norm", plot=pylab)


# In[46]:


plt.scatter(startup_pred,final_ml.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


# In[48]:


from sklearn.model_selection import train_test_split
startup_train,startup_test  = train_test_split(startup_new,test_size = 0.2)
startup_train
startup_test


# In[49]:


model_train = smf.ols("ms~Pr+rd+ad",data=startup_train).fit()


# In[50]:


train_pred = model_train.predict(startup_train)


# In[51]:


train_resid  = train_pred - startup_train.Pr


# In[52]:


train_resid


# In[53]:


train_rmse = np.sqrt(np.mean(train_resid*train_resid))
train_rmse


# In[54]:


test_pred = model_train.predict(startup_test)
test_pred


# In[55]:


test_resid  = test_pred - startup_test.Pr


# In[56]:


test_rmse = np.sqrt(np.mean(test_resid*test_resid))
test_rmse


# In[ ]:




