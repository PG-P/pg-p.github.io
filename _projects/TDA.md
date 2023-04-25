```python
import numpy as np
import pandas as pd
import pickle as pickle
import gudhi as gd  
from pylab import *
%matplotlib inline
```

The next statments load the  correlation matrices with pandas:


```python
path_file = "/Users/piergiorgiopanero/Desktop/corr/"
files_list = [
    '1anf.corr_1.txt', 
    '1ez9.corr_1.txt', 
    '1fqa.corr_2.txt', 
    '1fqb.corr_3.txt', 
    '1fqc.corr_2.txt', 
    '1fqd.corr_3.txt', 
    '1jw4.corr_4.txt', 
    '1jw5.corr_5.txt', 
    '1lls.corr_6.txt', 
    '1mpd.corr_4.txt', 
    '1omp.corr_7.txt', 
    '3hpi.corr_5.txt', 
    '3mbp.corr_6.txt', 
    '4mbp.corr_7.txt'
]
corr_list = [
    pd.read_csv(
        path_file + u, 
        header = None, 
        delim_whitespace = True
    ) for u in files_list
]
```

The object corr_list is a list of  correlation matrices. We can iterate in the list to compute the matrix of distances associated to each configuration:


```python
dist_list = [1 - np.abs(c) for c in corr_list]
```

Let's print out the first lines of the first distance matrix:


```python
D = dist_list[0]
D.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>360</th>
      <th>361</th>
      <th>362</th>
      <th>363</th>
      <th>364</th>
      <th>365</th>
      <th>366</th>
      <th>367</th>
      <th>368</th>
      <th>369</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.076200</td>
      <td>0.171364</td>
      <td>0.378207</td>
      <td>0.461747</td>
      <td>0.493499</td>
      <td>0.478665</td>
      <td>0.432338</td>
      <td>0.568455</td>
      <td>0.639504</td>
      <td>...</td>
      <td>0.694159</td>
      <td>0.723059</td>
      <td>0.660802</td>
      <td>0.614051</td>
      <td>0.660601</td>
      <td>0.686334</td>
      <td>0.640850</td>
      <td>0.617944</td>
      <td>0.695108</td>
      <td>0.748451</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.076200</td>
      <td>0.000000</td>
      <td>0.122763</td>
      <td>0.233837</td>
      <td>0.350744</td>
      <td>0.406213</td>
      <td>0.425202</td>
      <td>0.381799</td>
      <td>0.541636</td>
      <td>0.646580</td>
      <td>...</td>
      <td>0.817461</td>
      <td>0.844610</td>
      <td>0.781266</td>
      <td>0.740222</td>
      <td>0.793586</td>
      <td>0.808770</td>
      <td>0.754748</td>
      <td>0.730646</td>
      <td>0.804961</td>
      <td>0.848953</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.171364</td>
      <td>0.122763</td>
      <td>0.000000</td>
      <td>0.084642</td>
      <td>0.131528</td>
      <td>0.148980</td>
      <td>0.162259</td>
      <td>0.164105</td>
      <td>0.333175</td>
      <td>0.480605</td>
      <td>...</td>
      <td>0.782234</td>
      <td>0.813481</td>
      <td>0.718610</td>
      <td>0.666239</td>
      <td>0.742311</td>
      <td>0.740322</td>
      <td>0.667525</td>
      <td>0.653546</td>
      <td>0.742430</td>
      <td>0.766030</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.378207</td>
      <td>0.233837</td>
      <td>0.084642</td>
      <td>0.000000</td>
      <td>0.045478</td>
      <td>0.132107</td>
      <td>0.217299</td>
      <td>0.293902</td>
      <td>0.494307</td>
      <td>0.675635</td>
      <td>...</td>
      <td>0.913930</td>
      <td>0.929677</td>
      <td>0.853475</td>
      <td>0.813717</td>
      <td>0.868041</td>
      <td>0.856491</td>
      <td>0.797437</td>
      <td>0.781044</td>
      <td>0.842591</td>
      <td>0.858435</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.461747</td>
      <td>0.350744</td>
      <td>0.131528</td>
      <td>0.045478</td>
      <td>0.000000</td>
      <td>0.059280</td>
      <td>0.146850</td>
      <td>0.297257</td>
      <td>0.511751</td>
      <td>0.734412</td>
      <td>...</td>
      <td>0.967195</td>
      <td>0.979578</td>
      <td>0.889375</td>
      <td>0.841519</td>
      <td>0.895967</td>
      <td>0.874522</td>
      <td>0.804538</td>
      <td>0.779865</td>
      <td>0.841695</td>
      <td>0.849836</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 370 columns</p>
</div>



The RipsComplex() function creates a 1-skeleton from the point cloud


```python
skeleton_protein = gd.RipsComplex(
    distance_matrix = D.values, 
    max_edge_length = 0.8
)
```

The max_edge_length parameter is the maximal diameter: only the edges of length less vers this value are included in the one skeleton graph.

Next, we create the Rips simplicial complex from this one-skeleton graph. This is a filtered Rips complex which filtration function is exacly the diameter of the simplices. We use the create_simplex_tree() function:


```python
Rips_simplex_tree_protein = skeleton_protein.create_simplex_tree(max_dimension = 2)
```

The max_dimension parameter is the maximum dimension of the simplices included in the the filtration. The object returned by the function is a simplex tree, of dimension 2 in this example:


```python
Rips_simplex_tree_protein.dimension()
```




    2



The number of simplices in a Rips complex increases very fast with the number of points and the dimension. There is more than on million of simplexes in the Rips complex:


```python
Rips_simplex_tree_protein.num_simplices()
```




    1626660



Note that this is actually the number of simplices in the "last" Rips complex of the filtration, namely with parameter  max_edge_length=0.8.


Now we can compute persistence on the simplex tree structure using the persistence() method of the simplex tree class:


```python
BarCodes_Rips = Rips_simplex_tree_protein.persistence()
```

The object BarCodes_Rips is the list of barcodes: each element in the list is a tuple (dim,(b,d)) where dim is a dimension, b is birth parameter and d is death parameter.

Let's print the 20 first elements in the list:


```python
for i in range(20):
    print(BarCodes_Rips[i])
```

    (1, (0.07963602000000003, 0.35798637))
    (1, (0.12677510000000003, 0.39508646999999997))
    (1, (0.26003449999999995, 0.5273952))
    (1, (0.07943339999999999, 0.31429881000000004))
    (1, (0.08248586999999996, 0.30429980999999995))
    (1, (0.11378021999999999, 0.31171713999999995))
    (1, (0.07726765000000002, 0.26078758))
    (1, (0.09107215000000002, 0.25065161999999996))
    (1, (0.0709843, 0.22765623000000001))
    (1, (0.09347402000000005, 0.24999733000000002))
    (1, (0.07013614000000001, 0.22504734000000004))
    (1, (0.08752541000000003, 0.20355559))
    (1, (0.21541215000000002, 0.32814707))
    (1, (0.06835270000000004, 0.17527247))
    (1, (0.08857625000000002, 0.19539684000000002))
    (1, (0.08241111999999995, 0.18353136999999997))
    (1, (0.10362273, 0.20264340000000003))
    (1, (0.09289979999999998, 0.19181444000000003))
    (1, (0.09581541999999998, 0.19310879999999997))
    (1, (0.09541275000000005, 0.18175165000000004))


These 20 topolological features have dimension 1, they corresponds to holes of dimension 1.

We have access to persistence_intervals per dimension using the persistence_intervals_in_dimension() method, for instance for dimension 0:


```python
Rips_simplex_tree_protein.persistence_intervals_in_dimension(0)
```




    array([[0.        , 0.01498817],
           [0.        , 0.01614179],
           [0.        , 0.01618629],
           [0.        , 0.01964641],
           [0.        , 0.02002598],
           [0.        , 0.02019621],
           [0.        , 0.02128167],
           [0.        , 0.02139837],
           [0.        , 0.02189414],
           [0.        , 0.02211046],
           [0.        , 0.02279135],
           [0.        , 0.02305466],
           [0.        , 0.02376384],
           [0.        , 0.02398958],
           [0.        , 0.02427977],
           [0.        , 0.02431161],
           [0.        , 0.02514183],
           [0.        , 0.0252621 ],
           [0.        , 0.025566  ],
           [0.        , 0.02572414],
           [0.        , 0.02668906],
           [0.        , 0.02703624],
           [0.        , 0.02723942],
           [0.        , 0.02730715],
           [0.        , 0.02737215],
           [0.        , 0.02743006],
           [0.        , 0.02758413],
           [0.        , 0.02769299],
           [0.        , 0.0277112 ],
           [0.        , 0.02815389],
           [0.        , 0.02889452],
           [0.        , 0.0291826 ],
           [0.        , 0.02937602],
           [0.        , 0.02954253],
           [0.        , 0.02960991],
           [0.        , 0.02965513],
           [0.        , 0.02989476],
           [0.        , 0.0299387 ],
           [0.        , 0.03005481],
           [0.        , 0.03019355],
           [0.        , 0.0302212 ],
           [0.        , 0.03044412],
           [0.        , 0.03061732],
           [0.        , 0.03066776],
           [0.        , 0.03093647],
           [0.        , 0.03093904],
           [0.        , 0.03096135],
           [0.        , 0.03119385],
           [0.        , 0.03127303],
           [0.        , 0.03137071],
           [0.        , 0.03139056],
           [0.        , 0.03141663],
           [0.        , 0.03147105],
           [0.        , 0.03165437],
           [0.        , 0.03197555],
           [0.        , 0.0320141 ],
           [0.        , 0.03203755],
           [0.        , 0.0320416 ],
           [0.        , 0.03204383],
           [0.        , 0.03205409],
           [0.        , 0.03209123],
           [0.        , 0.03216506],
           [0.        , 0.03251925],
           [0.        , 0.03252582],
           [0.        , 0.0326297 ],
           [0.        , 0.03320018],
           [0.        , 0.03320555],
           [0.        , 0.03322223],
           [0.        , 0.03329767],
           [0.        , 0.03348698],
           [0.        , 0.03385555],
           [0.        , 0.03414255],
           [0.        , 0.03438535],
           [0.        , 0.03452845],
           [0.        , 0.03464101],
           [0.        , 0.0351205 ],
           [0.        , 0.03512744],
           [0.        , 0.03513682],
           [0.        , 0.03527348],
           [0.        , 0.03541263],
           [0.        , 0.03559076],
           [0.        , 0.03564263],
           [0.        , 0.03568637],
           [0.        , 0.03572321],
           [0.        , 0.03593942],
           [0.        , 0.03595195],
           [0.        , 0.03602799],
           [0.        , 0.03609681],
           [0.        , 0.03610733],
           [0.        , 0.03616306],
           [0.        , 0.03644183],
           [0.        , 0.03654824],
           [0.        , 0.03661895],
           [0.        , 0.03724071],
           [0.        , 0.03733498],
           [0.        , 0.03768468],
           [0.        , 0.03774493],
           [0.        , 0.03798143],
           [0.        , 0.03815909],
           [0.        , 0.03847974],
           [0.        , 0.03851494],
           [0.        , 0.03857502],
           [0.        , 0.03892696],
           [0.        , 0.03907595],
           [0.        , 0.03913963],
           [0.        , 0.0392337 ],
           [0.        , 0.03926192],
           [0.        , 0.03947112],
           [0.        , 0.03964138],
           [0.        , 0.0397164 ],
           [0.        , 0.03975846],
           [0.        , 0.03984852],
           [0.        , 0.03997199],
           [0.        , 0.04015835],
           [0.        , 0.04017049],
           [0.        , 0.04023692],
           [0.        , 0.04046704],
           [0.        , 0.04072276],
           [0.        , 0.04074775],
           [0.        , 0.04088823],
           [0.        , 0.04095619],
           [0.        , 0.04096195],
           [0.        , 0.04110269],
           [0.        , 0.04123369],
           [0.        , 0.04127687],
           [0.        , 0.04131072],
           [0.        , 0.04140789],
           [0.        , 0.04165062],
           [0.        , 0.04176154],
           [0.        , 0.04181859],
           [0.        , 0.04191496],
           [0.        , 0.04196476],
           [0.        , 0.0420939 ],
           [0.        , 0.04227029],
           [0.        , 0.04236735],
           [0.        , 0.04254986],
           [0.        , 0.04260688],
           [0.        , 0.04261734],
           [0.        , 0.04266441],
           [0.        , 0.04296581],
           [0.        , 0.04303701],
           [0.        , 0.04306739],
           [0.        , 0.04317269],
           [0.        , 0.04346509],
           [0.        , 0.0437037 ],
           [0.        , 0.04372465],
           [0.        , 0.04383413],
           [0.        , 0.04391404],
           [0.        , 0.04398658],
           [0.        , 0.04400293],
           [0.        , 0.04407297],
           [0.        , 0.04409098],
           [0.        , 0.04435811],
           [0.        , 0.04439728],
           [0.        , 0.04443404],
           [0.        , 0.0444774 ],
           [0.        , 0.04451368],
           [0.        , 0.04482899],
           [0.        , 0.04495304],
           [0.        , 0.04547506],
           [0.        , 0.04547828],
           [0.        , 0.0455478 ],
           [0.        , 0.04615431],
           [0.        , 0.04619047],
           [0.        , 0.04625533],
           [0.        , 0.04628039],
           [0.        , 0.04678061],
           [0.        , 0.04679402],
           [0.        , 0.04680115],
           [0.        , 0.04704526],
           [0.        , 0.04719844],
           [0.        , 0.04747816],
           [0.        , 0.04767326],
           [0.        , 0.04768366],
           [0.        , 0.04792482],
           [0.        , 0.04797942],
           [0.        , 0.04802758],
           [0.        , 0.04811274],
           [0.        , 0.04824104],
           [0.        , 0.04824237],
           [0.        , 0.04833753],
           [0.        , 0.04848755],
           [0.        , 0.0485553 ],
           [0.        , 0.04857819],
           [0.        , 0.04860536],
           [0.        , 0.04885771],
           [0.        , 0.04897734],
           [0.        , 0.04901851],
           [0.        , 0.04944188],
           [0.        , 0.04948698],
           [0.        , 0.04950527],
           [0.        , 0.049558  ],
           [0.        , 0.04972988],
           [0.        , 0.04981983],
           [0.        , 0.04982386],
           [0.        , 0.04984923],
           [0.        , 0.04986276],
           [0.        , 0.05027105],
           [0.        , 0.0504428 ],
           [0.        , 0.05055695],
           [0.        , 0.05066297],
           [0.        , 0.05070345],
           [0.        , 0.0509538 ],
           [0.        , 0.05103037],
           [0.        , 0.0510853 ],
           [0.        , 0.0511247 ],
           [0.        , 0.05119141],
           [0.        , 0.05125808],
           [0.        , 0.0514161 ],
           [0.        , 0.05146225],
           [0.        , 0.05157315],
           [0.        , 0.05181604],
           [0.        , 0.0518984 ],
           [0.        , 0.05228542],
           [0.        , 0.05245802],
           [0.        , 0.05249016],
           [0.        , 0.05256426],
           [0.        , 0.05261345],
           [0.        , 0.052687  ],
           [0.        , 0.05269135],
           [0.        , 0.05286959],
           [0.        , 0.05308782],
           [0.        , 0.05312981],
           [0.        , 0.05328802],
           [0.        , 0.05388745],
           [0.        , 0.05406629],
           [0.        , 0.05424359],
           [0.        , 0.05432115],
           [0.        , 0.05434062],
           [0.        , 0.05447437],
           [0.        , 0.05475445],
           [0.        , 0.0547687 ],
           [0.        , 0.05480814],
           [0.        , 0.05501963],
           [0.        , 0.05515901],
           [0.        , 0.0551603 ],
           [0.        , 0.05521654],
           [0.        , 0.05539072],
           [0.        , 0.05547035],
           [0.        , 0.05559308],
           [0.        , 0.05608519],
           [0.        , 0.05611108],
           [0.        , 0.05611354],
           [0.        , 0.05622496],
           [0.        , 0.05627249],
           [0.        , 0.05637189],
           [0.        , 0.057297  ],
           [0.        , 0.05729971],
           [0.        , 0.05746997],
           [0.        , 0.05753706],
           [0.        , 0.05846568],
           [0.        , 0.05849421],
           [0.        , 0.05855491],
           [0.        , 0.05858433],
           [0.        , 0.05881015],
           [0.        , 0.05891398],
           [0.        , 0.05910734],
           [0.        , 0.05927985],
           [0.        , 0.05930276],
           [0.        , 0.05937792],
           [0.        , 0.0595791 ],
           [0.        , 0.05959093],
           [0.        , 0.05960105],
           [0.        , 0.06001639],
           [0.        , 0.06091708],
           [0.        , 0.06102111],
           [0.        , 0.0611472 ],
           [0.        , 0.061409  ],
           [0.        , 0.06150065],
           [0.        , 0.06153714],
           [0.        , 0.06154331],
           [0.        , 0.06164122],
           [0.        , 0.0619127 ],
           [0.        , 0.06200894],
           [0.        , 0.06212457],
           [0.        , 0.0624782 ],
           [0.        , 0.06275173],
           [0.        , 0.06289645],
           [0.        , 0.06309263],
           [0.        , 0.06329755],
           [0.        , 0.06339085],
           [0.        , 0.06340604],
           [0.        , 0.0634087 ],
           [0.        , 0.06365607],
           [0.        , 0.06370852],
           [0.        , 0.06376314],
           [0.        , 0.06452474],
           [0.        , 0.06464462],
           [0.        , 0.06539584],
           [0.        , 0.0655788 ],
           [0.        , 0.06560913],
           [0.        , 0.06565494],
           [0.        , 0.06628835],
           [0.        , 0.0664005 ],
           [0.        , 0.06651979],
           [0.        , 0.0665555 ],
           [0.        , 0.06667502],
           [0.        , 0.0667084 ],
           [0.        , 0.06679813],
           [0.        , 0.0668217 ],
           [0.        , 0.06688167],
           [0.        , 0.06728802],
           [0.        , 0.06736142],
           [0.        , 0.06764264],
           [0.        , 0.06825241],
           [0.        , 0.06869392],
           [0.        , 0.06877336],
           [0.        , 0.06892966],
           [0.        , 0.06897429],
           [0.        , 0.06918575],
           [0.        , 0.06954789],
           [0.        , 0.06996146],
           [0.        , 0.07021497],
           [0.        , 0.07103847],
           [0.        , 0.07105814],
           [0.        , 0.07122811],
           [0.        , 0.07162656],
           [0.        , 0.07171408],
           [0.        , 0.07273788],
           [0.        , 0.07326412],
           [0.        , 0.07355539],
           [0.        , 0.07385251],
           [0.        , 0.0738902 ],
           [0.        , 0.07394715],
           [0.        , 0.07403273],
           [0.        , 0.0746566 ],
           [0.        , 0.07496114],
           [0.        , 0.07510989],
           [0.        , 0.07526754],
           [0.        , 0.0761947 ],
           [0.        , 0.07620041],
           [0.        , 0.07628743],
           [0.        , 0.07870332],
           [0.        , 0.07881973],
           [0.        , 0.07893213],
           [0.        , 0.07893759],
           [0.        , 0.07933292],
           [0.        , 0.08004957],
           [0.        , 0.08036911],
           [0.        , 0.08080199],
           [0.        , 0.08109694],
           [0.        , 0.081755  ],
           [0.        , 0.08176781],
           [0.        , 0.08248695],
           [0.        , 0.08258634],
           [0.        , 0.08299805],
           [0.        , 0.08464198],
           [0.        , 0.08729047],
           [0.        , 0.08744106],
           [0.        , 0.08878104],
           [0.        , 0.08969459],
           [0.        , 0.09321352],
           [0.        , 0.09341871],
           [0.        , 0.09479173],
           [0.        , 0.09541855],
           [0.        , 0.10237913],
           [0.        , 0.10285434],
           [0.        , 0.10715871],
           [0.        , 0.10830089],
           [0.        , 0.10954903],
           [0.        , 0.11366507],
           [0.        , 0.11958183],
           [0.        , 0.14547953],
           [0.        , 0.18964299],
           [0.        , 0.39963213],
           [0.        , 0.40540978],
           [0.        , 0.4251596 ],
           [0.        , 0.50723615],
           [0.        , 0.66869636],
           [0.        ,        inf]])



The last bars (0.0, inf) die at infinity.

Finally we can plot the points (birth, death) in the so-called persistence diagram:


```python
gd.plot_persistence_diagram(BarCodes_Rips);
```


    
![png](output_22_0.png)
    


In this representation, -dimensional features are the red points (holes of dimension , namely connected components). The last connected component dies at infinity in the filtration (red point at the top). The -dimensional features are represented in blue.

The most persistent topological features are those points that are far from the diagonal. Further in the tutorial we give statistical methods to identify significant topological features.

Note that this representation does not say which points are "at the origin" of a given feature. Moreover, a given topological feature (namely a homology class that corresponds to a hole of dimension ) is by definition a class of cycles defined on the point cloud and thus it can be represented by several cycles.

**Bottleneck distance**

To exploit the topological information and topological features inferred from persistent homology, one needs to be able to compare persistence diagrams.

We see a persistence diagram as the union of its points and of the diagonal, where the point of the diagonal are counted with infinite multiplicity.
  	 
Let us compute the Rips complex filtration for another configuration of protein:


```python
D1 = dist_list[1]

skeleton_protein1 = gd.RipsComplex(
    distance_matrix = D1.values, 
    max_edge_length = 0.8
) 

Rips_simplex_tree_protein1 = skeleton_protein1.create_simplex_tree(max_dimension = 2)
```

and the barcode for this filtration:


```python
BarCodes_Rips1 = Rips_simplex_tree_protein1.persistence()
```

The bottleneck distance between the two persistence diagrams can be computed using the bottleneck_distance() function. The bottleneck distance is computed per dimension (for dimension 1 in the example below). We can give in argument of the function the persistence_intervals for a given dimension, which can be computed using the persistence_intervals_in_dimension() function.


```python
I0 = Rips_simplex_tree_protein.persistence_intervals_in_dimension(1)
I1 = Rips_simplex_tree_protein1.persistence_intervals_in_dimension(1)

gd.bottleneck_distance(I0, I1)
```




    0.05052142999999998




```python
gd.plot_persistence_diagram(BarCodes_Rips1);
```


    
![png](output_30_0.png)
    



```python

```
