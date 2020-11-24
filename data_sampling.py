"""
    total : 494,021     => 1,000개 기준
    neptune : 107,201   => 210개
    smurf : 280,790     => 560개
    normal : 97,278     => 230개
    한번 시험해보자
"""

import pandas as pd

column_list = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment',
               'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
               'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login',
               'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
               'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count',
               'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
               'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
               'dst_host_rerror_rate','dst_host_srv_rerror_rate','class']

train_data = pd.read_csv('F:/data/harmony_search/traindata.txt', header=None)

train_data.columns = column_list
train_data = train_data.drop(['su_attempted','wrong_fragment','urgent','root_shell','num_root','num_failed_logins','num_compromised','num_access_files','is_guest_login','hot'], axis=1)
'''
train_data = train_data.drop(['dst_host_serror_rate','dst_host_srv_serror_rate','dst_bytes','serror_rate','srv_serror_rate',
                  'logged_in','dst_host_count','dst_host_srv_diff_host_rate','srv_diff_host_rate',
                  'dst_host_srv_rerror_rate','duration','srv_rerror_rate','dst_host_rerror_rate','rerror_rate',
                  'num_shells','land','num_outbound_cmds','is_host_login','num_file_creations','su_attempted',
                  'wrong_fragment','urgent','root_shell','num_root','num_failed_logins','num_compromised',
                  'num_access_files','is_guest_login','hot'], axis=1)
'''
print(train_data.head())

print(set(train_data['class']))

neptune_data = train_data[train_data['class']=='neptune.']
neptune_data = neptune_data.sample(n=210, random_state=42)
smurf_data = train_data[train_data['class']=='smurf.']
smurf_data = smurf_data.sample(n=560, random_state=42)
normal_data = train_data[train_data['class']=='normal.']
normal_data = normal_data.sample(n=230, random_state=42)

print(neptune_data.shape)
print(smurf_data.shape)
print(normal_data.shape)

#total_data = pd.concat([normal_data, smurf_data], axis=0, sort=False)
total_data = pd.concat([normal_data, neptune_data, smurf_data], axis=0, sort=False)
print(total_data.shape)
print(total_data.head())

total_data.to_csv('total_data.csv', index=False)