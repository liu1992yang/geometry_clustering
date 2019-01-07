%mem=64gb
%nproc=28       
%Chk=snap_40.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_40 

2     1 
  O    2.091102998  -4.317919193  -7.513289179
  C    2.153667224  -3.072660453  -8.229723033
  H    3.122145465  -3.152664142  -8.773196972
  H    2.218318982  -2.243575047  -7.489404918
  C    0.999826191  -2.856324819  -9.207493424
  H    1.046999391  -1.823903281  -9.640131039
  O    1.211946323  -3.720717602 -10.374295837
  C    0.049236095  -4.478268695 -10.670431067
  H   -0.046082739  -4.497827572 -11.784969974
  C   -0.425438493  -3.185303586  -8.688714198
  H   -0.431801168  -3.843718874  -7.779670978
  C   -1.112910629  -3.876502656  -9.874207067
  H   -1.675534540  -3.128908992 -10.478769414
  H   -1.870050216  -4.617610465  -9.559580158
  O   -1.159416237  -1.964156858  -8.440401361
  N    0.334846736  -5.891002232 -10.212124971
  C   -0.178617260  -7.070899277 -10.791371984
  C    0.336391156  -8.156463529 -10.030401011
  N    1.143529746  -7.617687188  -9.006728171
  C    1.135911922  -6.267154195  -9.124214768
  N   -1.009998018  -7.174860097 -11.872014917
  C   -1.331048887  -8.465965712 -12.221701740
  N   -0.848700428  -9.599941544 -11.532968521
  C    0.027723323  -9.526168549 -10.364138049
  N   -2.193214078  -8.640096982 -13.274951435
  H   -2.511389249  -7.826471051 -13.796961979
  H   -2.392319625  -9.545967996 -13.675075714
  O    0.367061917 -10.558360007  -9.849183249
  H    1.688572076  -5.485434381  -8.447026314
  H   -1.111894849 -10.549204016 -11.834788155
  P   -0.833394568  -1.319791374  -6.956397350
  O   -2.156231249  -0.359226950  -6.684796212
  C   -1.886724532   1.044956179  -6.515810788
  H   -2.883994010   1.500973847  -6.692794174
  H   -1.174252130   1.430122860  -7.272084128
  C   -1.385085396   1.287578078  -5.081750296
  H   -0.299514201   1.040412447  -4.934838399
  O   -1.426719825   2.724767623  -4.827813288
  C   -2.447796847   3.056759983  -3.879699357
  H   -1.936579410   3.681381689  -3.100473181
  C   -2.285787280   0.617903613  -4.014340704
  H   -2.938131943  -0.184388251  -4.433191913
  C   -3.090602248   1.767128558  -3.374499615
  H   -3.028744091   1.694070677  -2.262875524
  H   -4.166324344   1.669404255  -3.612228471
  O   -1.366767071   0.128918849  -3.027330144
  O    0.454154468  -0.615796333  -6.837234886
  O   -1.125319020  -2.588072654  -6.019686347
  N   -3.398145695   3.943945584  -4.635781152
  C   -3.041169599   4.780526312  -5.688666518
  C   -4.203716595   5.546705367  -6.025689504
  N   -5.262708244   5.179565892  -5.183880678
  C   -4.784859191   4.241867934  -4.364617282
  N   -1.835268204   4.947107384  -6.368385332
  C   -1.827916264   5.922827861  -7.380628374
  N   -2.874175746   6.658712760  -7.722786165
  C   -4.119184877   6.522586585  -7.070902772
  H   -0.870242924   6.079394193  -7.927200379
  N   -5.133564491   7.309091546  -7.462817718
  H   -5.018087747   7.998407742  -8.207000851
  H   -6.048513867   7.251467855  -7.012846560
  H   -5.333371969   3.746064318  -3.568236682
  P   -1.535197592  -1.413955252  -2.465660647
  O   -0.798913646  -1.137325365  -1.045541396
  C   -0.158964467  -2.298746464  -0.447934367
  H    0.588466226  -2.735281154  -1.151221781
  H    0.361913246  -1.858447593   0.427990605
  C   -1.272444830  -3.263799466  -0.045894446
  H   -1.896540706  -2.885304174   0.797708787
  O   -2.179888930  -3.269655157  -1.207752623
  C   -2.426418846  -4.650703398  -1.658856391
  H   -3.536781589  -4.670867384  -1.781548162
  C   -0.838510529  -4.731531583   0.181512403
  H    0.205799446  -4.938313334  -0.175365833
  C   -1.889174689  -5.560463350  -0.566049346
  H   -1.463538075  -6.518323031  -0.975276224
  H   -2.688642333  -5.898144218   0.126180266
  O   -0.964581593  -4.919553505   1.604139559
  O   -0.964228916  -2.388451158  -3.440042491
  O   -3.163755118  -1.401851838  -2.297223626
  N   -1.772591026  -4.850166378  -2.974953202
  C   -0.342110135  -4.794728278  -3.126337726
  N    0.145743820  -5.056581485  -4.438890652
  C   -0.668220906  -5.285158364  -5.570637660
  C   -2.115191884  -5.230096353  -5.348263567
  C   -2.608639047  -5.015636310  -4.102025390
  O    0.426230909  -4.547507075  -2.218847657
  H    1.172142234  -5.140272525  -4.523183804
  O   -0.096706916  -5.478331380  -6.648123866
  C   -2.983031195  -5.353310119  -6.552180511
  H   -3.976975896  -5.763014195  -6.326751021
  H   -3.128866574  -4.364685118  -7.021823915
  H   -2.527439809  -6.015677840  -7.305923352
  H   -3.689716271  -4.973065662  -3.914170343
  P   -0.022063234  -6.104383460   2.263266566
  O    1.360736925  -5.790673737   1.450233631
  C    2.628564082  -6.150871520   2.073538959
  H    2.768608847  -5.506191610   2.964715525
  H    2.618484111  -7.221840384   2.360336533
  C    3.701452849  -5.891080563   1.004675215
  H    4.660254941  -5.557485377   1.468393946
  O    4.058583451  -7.158492942   0.394586817
  C    3.493444670  -7.261287694  -0.931140201
  H    4.231047756  -7.873196992  -1.500960943
  C    3.258391882  -4.954522410  -0.156532470
  H    2.268419158  -4.483217248   0.013684320
  C    3.303402863  -5.826489774  -1.419558131
  H    2.398357021  -5.675916499  -2.053852429
  H    4.161813264  -5.510016621  -2.053529514
  O    4.245942430  -3.938859406  -0.356199704
  O   -0.581868412  -7.458160049   2.267595199
  O    0.302606989  -5.445844899   3.723154920
  N    2.229575351  -8.063578768  -0.753128009
  C    0.946342816  -7.728642089  -1.375225256
  N   -0.206118914  -8.246915725  -0.806406776
  C   -0.141928214  -9.163049624   0.201703612
  C    1.126904735  -9.631524025   0.707571757
  C    2.268837685  -9.046957591   0.242049236
  O    0.934272690  -6.999704775  -2.355550301
  N   -1.332911972  -9.688709256   0.660753559
  H   -1.359027901 -10.080754581   1.594080937
  H   -2.194013896  -9.234966612   0.380506568
  H    1.275555918  -4.363295982  -6.925978741
  H    3.262788740  -9.316436473   0.636450963
  H    1.161677927 -10.425516452   1.449592571
  H    4.128984416  -3.194937716   0.266266416
  H   -1.100336895  -2.471466271  -4.949659789
  H   -3.567845033  -1.984063228  -1.581711740
  H   -0.264503451  -5.700635002   4.500201405
  H    1.616209419  -8.178623420  -8.305301748
  H   -1.019889401   4.366584844  -6.108762182
