import pyAudioAnalysis.audioFeatureExtraction as FE
from pyAudioAnalysis import audioBasicIO as IO
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import scale
from math import log
import itertools

def viterbi(observations, states, start_probability, transition_probability, emission_probability):  #Viterbi
    trellis = [{}]
    path = {}
    for state in states:
        trellis[0][state] = log(start_probability[state]) + log(emission_probability[state][observations[0]])
        path[state] = [state]
    for observations_index in range(1,len(observations)):
        trellis.append({})
        new_path = {}
        for state in states:
            (probability, possible_state) = max(
            [(trellis[observations_index-1][y0] + log(transition_probability[y0][state]) + log(emission_probability[state][observations[observations_index]]), y0) for y0 in states])
            trellis[observations_index][state] = probability
            new_path[state] = path[possible_state] + [state]
        path = new_path
    (probability, state) = max([(trellis[len(observations) - 1][state], state) for state in states])
    return probability, path[state]

states = ('p1', 'p2','silence')    #初始化Viterbi的概率
transition_probability = {
    'p1': {'p1':0.7, 'p2':0.15, 'silence':0.15},
    'p2': {'p1':0.15, 'p2':0.7, 'silence':0.15},
    'silence':{'p1':0.2, 'p2':0.2, 'silence':0.6}}
start_probability = {'p1': 0.4, 'p2': 0.4, 'silence':0.2}
emission_probability = {
    'p1': {'p1':0.7, 'p2':0.2,'silence':0.1},
    'p2': {'p1':0.2, 'p2':0.7, 'silence':0.1},
    'silence': {'p1':0.15, 'p2':0.15,'silence':0.7}
}

filepath = '/Users/tommy/Documents/GitHub/Intelligent-Sale/李萍萍-成交客户-曾璐燕-促成.wav'
[Fs, x] = IO.readAudioFile(filepath)
features = FE.stFeatureExtraction(x, Fs, 2000, 2000)  # load data and feature extraction

feature_rescale = scale(features[0].T)

labels = {}
for i in range(len(features[0][1])):
    if features[0][1][i] < 0.0001:  # energy < 0.0001 的就算silence
        labels[i] = 'silence'
    else:
        labels[i] = 'speech'

speech_feature = []  # 把语音的部分拿出来，kmeans来聚两类
index = []
for key in labels.keys():
    if labels[key] == 'speech':
        index.append(key)
        speech_feature.append(feature_rescale[key])
speech_feature = np.array(speech_feature)

kmeans = KMeans(n_clusters=2, max_iter=600)
clustering = kmeans.fit(speech_feature)

for i in range(len(index)):  # 把聚类结果贴标签
    if clustering.labels_[i] == 0:
        labels[index[i]] = 'p1'
    if clustering.labels_[i] == 1:
        labels[index[i]] = 'p2'

pro, newpath = viterbi(list(labels.values()), states, start_probability, transition_probability,
                       emission_probability)  # 用Viterbi smooth

speech_type = [k for k, v in itertools.groupby(newpath)]  # 把结果分组
speech_len = [len(list(v)) for k, v in itertools.groupby(newpath)]
time_interval = []
time_interval.append((0, speech_len[0] * 0.25))
for i in range(1, len(speech_len)):
    time_interval.append((time_interval[i - 1][1], time_interval[i - 1][1] + speech_len[i] * 0.25))

result = {}  # 把分组结果转化成时间
for i in range(len(time_interval)):
    result[time_interval[i]] = speech_type[i]

print(result)