from utilities import create_palette
from utilities import read_csv
from utilities import create_heatmap
from utilities import show_plot

# coding: utf-8

# # SVD Heatmaps

# In[67]:


languages = ['en','ru','de','es','fr','it', 'zh-CN']


# Create Palette

cmap = create_palette(10, 130, 20, True)
cmap1 = create_palette(130, 10, 20, True)


# In[69]:




s_min_df = read_csv("../HeatmapData/T/s_min.csv")
s_min_df = s_min_df.set_index([languages])
s_min_hm = create_heatmap(s_min_df,cmap)
title = s_min_hm.set_title('s_min')


# In[71]:


s_max_df = read_csv("../HeatmapData/T/s_max.csv")
s_max_df = s_max_df.set_index([languages])
s_max_hm = create_heatmap(s_max_df,cmap)
title = s_max_hm.set_title('s_max')


# In[72]:


s_mean_df = read_csv("../HeatmapData/T/s_mean.csv")
s_mean_df = s_mean_df.set_index([languages])
s_mean_hm = create_heatmap(s_mean_df,cmap)
title = s_mean_hm.set_title('s_mean')


# In[73]:


s_median_df = read_csv("../HeatmapData/T/s_median.csv")
s_meadian_df = s_median_df.set_index([languages])
s_median_hm = create_heatmap(s_median_df,cmap)
title = s_median_hm.set_title('s_median')


# In[ ]:


s_std_df = read_csv("../HeatmapData/T/s_std.csv")
s_std_df = s_std_df.set_index([languages])
s_std_hm = create_heatmap(s_std_df,cmap)
title = s_std_hm.set_title('s_std')


# In[ ]:





# In[86]:


s_fro_df = read_csv("../HeatmapData/T/s_fro.csv")
s_fro_df = s_fro_df.set_index([languages])
s_fro_hm = create_heatmap(s_fro_df,cmap1)
title = s_fro_hm.set_title('s_fro')


# In[79]:


s_acc_df = read_csv("../HeatmapData/T/s_acc.csv")
s_acc_df = s_acc_df.set_index([languages])
s_acc_hm = create_heatmap(s_acc_df,cmap)
title = s_acc_hm.set_title('s_acc')


# In[ ]:





# In[80]:


s1_min_df = read_csv("../HeatmapData/T/s1_min.csv")
s_min_df = s_min_df.set_index([languages])
s1_min_hm = create_heatmap(s1_min_df,cmap)
title = s1_min_hm.set_title('s1_min')


# In[81]:


s1_max_df = read_csv("../HeatmapData/T/s1_max.csv")
s_max_df = s_max_df.set_index([languages])
s1_max_hm = create_heatmap(s1_max_df,cmap)
title = s1_max_hm.set_title('s1_max')


# In[82]:


s1_mean_df = read_csv("../HeatmapData/T/s1_mean.csv")
s_mean_df = s_mean_df.set_index([languages])
s1_mean_hm = create_heatmap(s1_mean_df,cmap)
title = s1_mean_hm.set_title('s1_mean')


# In[83]:


s1_median_df = read_csv("../HeatmapData/T/s1_median.csv")
s_median_df = s_median_df.set_index([languages])
s1_median_hm = create_heatmap(s1_median_df,cmap)
title = s1_median_hm.set_title('s1_median')


# In[ ]:


s1_std_df = read_csv("../HeatmapData/T/s1_std.csv")
s1_std_df = s1_std_df.set_index([languages])
s1_std_hm = create_heatmap(s1_std_df,cmap)
title = s1_std_hm.set_title('s1_std')


# In[ ]:


show_plot()


# In[ ]:





# In[ ]:




