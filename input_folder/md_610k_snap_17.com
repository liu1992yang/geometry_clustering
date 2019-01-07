%mem=64gb
%nproc=28       
%Chk=snap_17.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_17 

2     1 
  O   -0.326663583  -6.375583011  -7.451797315
  C    0.729523700  -5.534647364  -6.982600166
  H    1.651170335  -6.005941522  -7.394301323
  H    0.773216591  -5.579830941  -5.873264911
  C    0.606707132  -4.087789377  -7.477502119
  H    1.187327897  -3.392483912  -6.817601859
  O    1.333326014  -3.975105288  -8.739793811
  C    0.453757769  -3.792694308  -9.834909359
  H    0.944908911  -3.028836275 -10.493906693
  C   -0.836400860  -3.596963884  -7.755528506
  H   -1.636780109  -4.258552054  -7.339765412
  C   -0.929347589  -3.435697806  -9.282728674
  H   -1.191276141  -2.374957903  -9.519871679
  H   -1.756906410  -4.027531955  -9.715317300
  O   -1.001892040  -2.240356515  -7.269031312
  N    0.439402773  -5.090922305 -10.608765774
  C    1.075671981  -5.312481420 -11.853043272
  C    0.937440080  -6.701329432 -12.139060705
  N    0.254554313  -7.299305898 -11.065030253
  C   -0.027030751  -6.335088008 -10.152523398
  N    1.695543197  -4.393463413 -12.646769401
  C    2.242402302  -4.908403122 -13.804780462
  N    2.149857089  -6.269579534 -14.164783758
  C    1.475444403  -7.279851979 -13.350244705
  N    2.897292682  -4.035379642 -14.630261393
  H    2.981362041  -3.056110474 -14.363908879
  H    3.351112356  -4.319377759 -15.487237279
  O    1.448887907  -8.408674994 -13.759634307
  H   -0.517537489  -6.468644651  -9.142493887
  H    2.568330246  -6.614935969 -15.042970772
  P   -1.309764268  -2.048917434  -5.671700094
  O   -1.525670493  -0.408450164  -5.677049002
  C   -0.679036964   0.344908478  -4.776085300
  H   -0.147310620   1.070596558  -5.433251063
  H    0.070603596  -0.294579612  -4.256378873
  C   -1.566616007   1.081022859  -3.769470168
  H   -1.068105458   1.136825452  -2.764362107
  O   -1.619450286   2.485295581  -4.187404969
  C   -2.954121044   2.925697353  -4.371724882
  H   -3.006857924   3.924039601  -3.880018701
  C   -3.035776719   0.592738080  -3.646473967
  H   -3.315575264  -0.197599070  -4.380569561
  C   -3.896458594   1.858530258  -3.815797093
  H   -4.298941412   2.150847233  -2.814627623
  H   -4.793383278   1.675229171  -4.435434848
  O   -3.283530399   0.153491116  -2.309778014
  O   -0.302110183  -2.569069043  -4.726467786
  O   -2.825535859  -2.574900936  -5.527387699
  N   -3.145876716   3.107942761  -5.854767969
  C   -3.461683171   4.279847269  -6.535864113
  C   -3.477236983   3.952667369  -7.932426863
  N   -3.151152378   2.598502298  -8.102119576
  C   -2.949285548   2.108239354  -6.879225013
  N   -3.754587158   5.574807728  -6.102288142
  C   -4.054311493   6.517364116  -7.107933634
  N   -4.081308771   6.253582155  -8.402138136
  C   -3.793955198   4.962363367  -8.894435100
  H   -4.286435838   7.553752716  -6.772276618
  N   -3.836607112   4.765911566 -10.222249532
  H   -4.066653143   5.523216567 -10.866339895
  H   -3.642376637   3.848140571 -10.621422170
  H   -2.658059682   1.081120113  -6.627649069
  P   -2.815567118  -1.394789166  -1.890664345
  O   -1.239941493  -0.942426558  -1.775811714
  C   -0.224727281  -1.949919140  -1.534983271
  H   -0.109260725  -2.576882947  -2.450973477
  H    0.691340778  -1.343295093  -1.386724007
  C   -0.635915422  -2.728362847  -0.288243205
  H   -0.672670392  -2.105516649   0.638181829
  O   -2.038046719  -3.070126825  -0.550328665
  C   -2.202861312  -4.517032316  -0.667029613
  H   -3.193078275  -4.692592175  -0.178505641
  C    0.107699823  -4.058714154  -0.029807989
  H    0.894568636  -4.272055703  -0.796114571
  C   -0.999170764  -5.118980569   0.028692537
  H   -0.676734497  -6.111390777  -0.386547741
  H   -1.226086486  -5.394751789   1.092957151
  O    0.681756589  -3.908781150   1.289321587
  O   -3.207839990  -2.412312436  -2.897420295
  O   -3.626054350  -1.406671976  -0.488598780
  N   -2.306412416  -4.873504685  -2.123059979
  C   -1.152183264  -5.124638684  -2.914490993
  N   -1.379442562  -5.480145639  -4.267010976
  C   -2.656080084  -5.482192530  -4.913190731
  C   -3.776044460  -5.138202111  -4.050850283
  C   -3.581578070  -4.833579145  -2.735855529
  O   -0.002360313  -5.077918304  -2.507157779
  H   -0.533320815  -5.689733737  -4.822077366
  O   -2.637776895  -5.734812302  -6.114690600
  C   -5.133799271  -5.119053117  -4.665613859
  H   -5.643735010  -4.157697211  -4.504544598
  H   -5.090655557  -5.276717860  -5.756877858
  H   -5.777338902  -5.914582534  -4.257243752
  H   -4.421115984  -4.549293626  -2.081904596
  P    2.116002971  -4.653440199   1.564003572
  O    2.159732438  -5.712525566   0.316822139
  C    3.443672711  -6.350496034   0.047449740
  H    4.122571891  -5.579261908  -0.367577083
  H    3.854745574  -6.787606657   0.982559805
  C    3.172001387  -7.468500208  -0.969378647
  H    3.973284722  -7.515651817  -1.742864159
  O    3.295813241  -8.729281131  -0.250210831
  C    2.008210208  -9.391113120  -0.185129038
  H    2.249947893 -10.476349729  -0.131777973
  C    1.759740228  -7.477860215  -1.620401671
  H    1.058067179  -6.748928405  -1.151850425
  C    1.277554702  -8.929506503  -1.444844985
  H    0.172312168  -9.007238443  -1.375756768
  H    1.572445697  -9.541357766  -2.321192590
  O    1.897617751  -7.171132354  -3.011395419
  O    2.363172927  -5.072932565   2.945167601
  O    3.182223267  -3.526164000   0.996955602
  N    1.394784997  -8.945628930   1.112994440
  C    0.249928319  -8.024186385   1.195954637
  N    0.122333462  -7.232583758   2.324239393
  C    0.931145532  -7.420127097   3.404448861
  C    1.949322619  -8.447403459   3.407961446
  C    2.184894699  -9.137943918   2.255449303
  O   -0.523377418  -7.961194306   0.251957992
  N    0.703649045  -6.633605343   4.514959845
  H    1.463211842  -6.494452023   5.170244489
  H    0.112373832  -5.815875610   4.399583445
  H   -1.146654603  -6.296613875  -6.855599149
  H    3.006337151  -9.865248764   2.175816446
  H    2.521848659  -8.644373510   4.311911623
  H    1.361561418  -6.363541177  -3.199143292
  H   -3.178199031  -2.736712186  -4.547107882
  H   -3.360895289  -0.746738092   0.216044272
  H    3.765907538  -3.066447556   1.650788691
  H    0.034676802  -8.291717627 -11.014862349
  H   -3.719859675   5.832022275  -5.112601710
