%mem=64gb
%nproc=28       
%Chk=snap_100.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_100 

2     1 
  O    2.831677817  -5.248885859  -6.725866658
  C    3.374374139  -3.997149668  -6.319719035
  H    4.407852691  -4.047593068  -6.731328781
  H    3.427477543  -3.967843762  -5.203244829
  C    2.643442820  -2.757904776  -6.838011152
  H    3.093996176  -1.822619950  -6.415173103
  O    2.901973310  -2.620999880  -8.272094196
  C    1.710980687  -2.845938016  -9.016554851
  H    1.808691390  -2.230161443  -9.943658343
  C    1.099280374  -2.753564426  -6.663744293
  H    0.697857784  -3.685656480  -6.206468033
  C    0.548542750  -2.513948315  -8.085313658
  H    0.271938131  -1.443697174  -8.199661370
  H   -0.385603407  -3.084013657  -8.279359388
  O    0.814346829  -1.587110493  -5.874974261
  N    1.775314455  -4.302721617  -9.433138423
  C    0.958717920  -5.385825990  -9.023074213
  C    1.689991154  -6.579109019  -9.311498336
  N    2.921933743  -6.214827731  -9.868432418
  C    2.980325210  -4.860923063  -9.919393992
  N   -0.279447172  -5.337924068  -8.462064126
  C   -0.808859810  -6.563262496  -8.111480157
  N   -0.110774536  -7.778920555  -8.293115087
  C    1.194217703  -7.881830057  -8.918228386
  N   -2.098875587  -6.628397178  -7.642962969
  H   -2.602968677  -5.745835111  -7.447071374
  H   -2.318289311  -7.364056880  -6.970871102
  O    1.704401027  -8.967578542  -9.018846841
  H    3.817197206  -4.254833451 -10.277578383
  H   -0.511251400  -8.654476811  -7.921770627
  P   -0.172546464  -1.715324581  -4.544330853
  O   -0.540271991  -0.105396939  -4.572413133
  C    0.077808539   0.766690594  -3.596981419
  H    0.766434456   1.418394916  -4.185531034
  H    0.653510410   0.201006295  -2.834354587
  C   -1.040380261   1.592455205  -2.963522363
  H   -0.796420621   1.855488500  -1.902086572
  O   -1.082793340   2.870333269  -3.677461177
  C   -2.433142509   3.327928857  -3.818004617
  H   -2.460471816   4.348131610  -3.356818426
  C   -2.480561754   1.020682073  -3.080730893
  H   -2.613164662   0.345902788  -3.963376861
  C   -3.354413848   2.279896284  -3.184268791
  H   -3.686262752   2.587995241  -2.162795716
  H   -4.298850144   2.107166235  -3.731120262
  O   -2.836039192   0.354593336  -1.878258631
  O    0.515243635  -2.311083953  -3.386025039
  O   -1.452445341  -2.306806915  -5.349706510
  N   -2.698492813   3.484734860  -5.282955297
  C   -2.640253865   2.541153404  -6.304577262
  C   -3.111056044   3.189332763  -7.495478251
  N   -3.434744269   4.523560387  -7.207147062
  C   -3.186374924   4.696232522  -5.911002958
  N   -2.257281657   1.203139171  -6.330797438
  C   -2.356349966   0.545176944  -7.564157342
  N   -2.793410918   1.098273740  -8.687800827
  C   -3.190073749   2.452471206  -8.719938696
  H   -2.061196023  -0.528374688  -7.572068062
  N   -3.622937563   2.961961047  -9.885813664
  H   -3.664774658   2.396771417 -10.732563941
  H   -3.924401128   3.934981437  -9.955013300
  H   -3.318265914   5.617972924  -5.346353889
  P   -2.494056724  -1.279307545  -1.828386897
  O   -1.371641675  -1.100446449  -0.667004069
  C   -0.527971407  -2.256070794  -0.401153760
  H   -0.058771637  -2.616789819  -1.342282033
  H    0.254868607  -1.839243610   0.272667663
  C   -1.418097261  -3.273899242   0.300047000
  H   -1.656426748  -2.992092928   1.353377064
  O   -2.688442400  -3.165592317  -0.429612625
  C   -3.230273875  -4.490038900  -0.732657106
  H   -4.262748892  -4.458593236  -0.301912528
  C   -0.993489531  -4.751638430   0.181655796
  H   -0.265697179  -4.929828299  -0.665802091
  C   -2.304882852  -5.510517453  -0.076761715
  H   -2.130835657  -6.422445292  -0.683137997
  H   -2.717964787  -5.887933950   0.887278808
  O   -0.486510759  -5.164943545   1.464007432
  O   -2.014108366  -1.826324726  -3.143372667
  O   -4.022744293  -1.685113000  -1.424458711
  N   -3.333677839  -4.578901708  -2.217465018
  C   -2.242527187  -5.004710656  -3.034650668
  N   -2.514871159  -5.098249658  -4.421790168
  C   -3.771163782  -4.778646717  -5.044505506
  C   -4.779475729  -4.218085326  -4.139776454
  C   -4.541936818  -4.126097288  -2.808848549
  O   -1.144571627  -5.296828243  -2.599913817
  H   -1.778389901  -5.550050161  -5.004819139
  O   -3.845428183  -4.961857908  -6.243571986
  C   -6.065255823  -3.794699572  -4.758108127
  H   -6.672650284  -4.672313182  -5.043657311
  H   -6.682815549  -3.176105025  -4.092664989
  H   -5.904507242  -3.218920881  -5.683997938
  H   -5.278201263  -3.690820653  -2.115062535
  P    1.161085639  -5.312661580   1.490031594
  O    1.204592079  -6.330225876   0.172902943
  C    2.432033891  -6.889039927  -0.301427616
  H    3.204839716  -6.952470755   0.490177169
  H    2.151800808  -7.926663705  -0.591139065
  C    2.901410215  -6.086963500  -1.526325106
  H    3.929734075  -5.674193996  -1.386093900
  O    3.043831740  -7.036711578  -2.633821495
  C    2.381140011  -6.571429645  -3.814837637
  H    3.117913609  -6.715220503  -4.644933781
  C    1.927281372  -4.998884502  -2.029213685
  H    0.885102713  -5.088321080  -1.609548121
  C    1.922431471  -5.139372581  -3.550027682
  H    0.916449537  -4.902831705  -3.960779112
  H    2.621688391  -4.403114480  -4.010513850
  O    2.492269060  -3.743055459  -1.613853155
  O    1.839684085  -5.713662913   2.704506586
  O    1.512113824  -3.854966067   0.821108585
  N    1.213473428  -7.506226712  -4.073062910
  C    0.184783254  -7.143030748  -5.024041866
  N   -0.844922794  -8.027641773  -5.283522501
  C   -0.876967583  -9.245639232  -4.646308554
  C    0.117405268  -9.605653731  -3.675196721
  C    1.139275039  -8.729213077  -3.414406269
  O    0.246801747  -6.047499110  -5.594053162
  N   -1.883982725 -10.095387358  -5.011269819
  H   -1.995023884 -11.003880774  -4.582880885
  H   -2.591539440  -9.813880737  -5.673676968
  H    1.867444858  -5.356420530  -6.519170333
  H    1.943443113  -8.954463144  -2.690384205
  H    0.067012703 -10.564262516  -3.157590013
  H    2.002303700  -2.996124262  -2.069797849
  H   -2.100867235  -2.879846299  -4.834812354
  H   -4.261939583  -1.613343372  -0.445941887
  H    2.238877939  -3.752642187   0.117295466
  H    3.630195251  -6.883342545 -10.163426719
  H   -1.800196287   0.741779063  -5.506008221
