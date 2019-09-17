# -*- coding: cp936 -*-
import Pycluster as pc
import numpy as np
import matplotlib.pylab as pl

 
def myCKDemo(filename,n):
    #������������ǻ�ȡ���ݣ����ھ������������λ�ڵ�3�͵�4�У���0��ʼ���㣩   
    data = np.loadtxt(filename, delimiter = "," ,usecols=(2,4,14,8))
    #��8�͵�9�У������˳��еľ�γ�����꣬�������ɢ��ͼ
    xy = np.loadtxt(filename, delimiter = "," ,usecols=(2,4))
    #clustermap�Ǿ���֮��ļ���,��¼ÿһ�����ݵ����id
    clustermap = pc.kcluster(data, n)[0]
    #centroids �Ƿ������֮��ľ�����������
    centroids = pc.clustercentroids(data, clusterid=clustermap)[0]
    #m�Ǿ������
    m = pc.distancematrix(data)
 
    #mass ������¼����ĵ����Ŀ
    mass = np.zeros(n)
    for c in clustermap: 
        mass[c] += 1 
   
   
    #sil������ϵͳ�������ڼ�¼ÿ���صĴ�С
    sil = np.zeros(n*len(data)) 
    sil.shape = ( len(data), n ) 
   
    for i in range( 0, len(data) ): 
        for j in range( i+1, len(data) ): 
            d = m[j][i] 
            sil[i, clustermap[j] ] += d 
            sil[j, clustermap[i] ] += d 
 
    for i in range(0,len(data)): 
        sil[i,:] /= mass 
   
    #s����ϵ����һ��������������Ч���Ĳ���
    #ֵ��-1 ���� 1֮�䣬ֵԽ�󣬱�ʾЧ��Խ�á�
    #С��0��˵���������Ԫ�ص�ƽ������С������������أ���ʾ����Ч�����á�
    #������1��˵������Ч���ȽϺá�
    s=0 
    for i in range( 0, len(data) ): 
        c = clustermap[i] 
        a = sil[i,c] 
        b = min(sil[i,range(0,c)+range(c+1,n)]) 
        si = (b-a)/max(b,a)
        s+=si 
   
    print n, s/len(data) 
   
    #ʹ��matplotlib����ɢ��ͼ��
    fig, ax = pl.subplots()
    #cmap���������ֲ�ͬ������ɫ
    cmap = pl.get_cmap('jet', n)
    cmap.set_under('gray')
    #xy�Ǿ�γ�ȣ���ҪΪ��ͨ����γ����������ͬ�����ڵ����ϵ�λ��
    x = [list(d)[0] for d in xy]   
    y = [list(d)[1] for d in xy] 
    cax = ax.scatter(x, y, c=clustermap, s=30, cmap=cmap, vmin=0, vmax=n)
    pl.show() 

if __name__ == '__main__':
    #filename������c2.txt���ڵ�·�����ĳ��Լ������ϵ�·������
    filename = r"d:\tmp\book\ML\pytest\c5.txt"
    #n��Ԥ��ֳɼ��ࡣ
    n = 10
    myCKDemo(filename,n) 
	
