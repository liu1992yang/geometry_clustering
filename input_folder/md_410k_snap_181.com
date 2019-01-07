%mem=64gb
%nproc=28       
%Chk=snap_181.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_181 

2     1 
  O    0.411669673  -1.713599507  -6.064422310
  C    0.531166333  -2.820983797  -6.953452207
  H    0.492038123  -3.772424050  -6.383288527
  H    1.549895070  -2.703469493  -7.379710597
  C   -0.542093361  -2.728913894  -8.047798784
  H   -0.624469310  -1.699170063  -8.462829369
  O   -0.063743095  -3.535934241  -9.177797937
  C   -0.951039967  -4.612224510  -9.448167124
  H   -0.923398203  -4.778510754 -10.552206283
  C   -1.928486473  -3.310548643  -7.665638753
  H   -1.912812436  -3.838026267  -6.682719793
  C   -2.296396389  -4.255443149  -8.818245180
  H   -2.955018060  -3.727872659  -9.549546223
  H   -2.905841863  -5.123991237  -8.498263863
  O   -2.898236391  -2.227454212  -7.686558790
  N   -0.322064784  -5.821922944  -8.776257024
  C   -0.929297073  -7.045440925  -8.427590225
  C    0.046390622  -7.779577878  -7.690838118
  N    1.217325523  -7.014155638  -7.613371462
  C    0.988225651  -5.837781257  -8.242024387
  N   -2.197956646  -7.482921155  -8.691943314
  C   -2.506645136  -8.713704997  -8.151084935
  N   -1.609937400  -9.472029207  -7.364848436
  C   -0.254165050  -9.053271190  -7.090370135
  N   -3.768970077  -9.201419322  -8.360375068
  H   -4.400866467  -8.710801813  -8.989582420
  H   -4.047161863 -10.126532495  -8.055601584
  O    0.427047706  -9.749211754  -6.370004732
  H    1.698271027  -5.007479821  -8.349086121
  H   -1.941808159 -10.305513594  -6.812398561
  P   -3.344753987  -1.727677275  -6.182376598
  O   -3.637471776  -0.119790076  -6.302839630
  C   -2.517103580   0.769234254  -6.470649630
  H   -3.027477587   1.759004693  -6.466824068
  H   -2.059127570   0.608153226  -7.469193223
  C   -1.475530368   0.660548453  -5.348576059
  H   -0.750431439  -0.191043765  -5.521760189
  O   -0.650592002   1.866912365  -5.456867156
  C   -0.219355679   2.249665777  -4.128844017
  H    0.800228153   1.802341403  -3.973211071
  C   -2.032596399   0.635083548  -3.900395402
  H   -3.143360744   0.719708429  -3.852990625
  C   -1.297443321   1.761351208  -3.155275821
  H   -0.857698608   1.368051709  -2.206548621
  H   -1.995136172   2.562437416  -2.841079701
  O   -1.556595796  -0.571291742  -3.265756487
  O   -2.129115774  -2.037457950  -5.303496175
  O   -4.786275664  -2.413577546  -6.047057209
  N   -0.086998603   3.732432268  -4.154221387
  C   -0.206256478   4.594343437  -5.235961963
  C    0.131323880   5.904019829  -4.759076505
  N    0.452425567   5.838206884  -3.395872449
  C    0.328346696   4.561906188  -3.040289976
  N   -0.569687674   4.375031026  -6.562959069
  C   -0.577260220   5.508731183  -7.397601287
  N   -0.268976193   6.732586463  -7.003193820
  C    0.116799721   7.007360871  -5.671902360
  H   -0.871526129   5.353211601  -8.459552484
  N    0.448350763   8.267355643  -5.353908617
  H    0.426091915   9.017347159  -6.046433703
  H    0.735256114   8.515120470  -4.405844742
  H    0.504997521   4.138022781  -2.053992302
  P   -2.591244216  -1.852976363  -3.174205274
  O   -1.487958204  -3.045659008  -3.238204783
  C   -1.962991949  -4.390858752  -2.972920012
  H   -2.990901415  -4.564596996  -3.346498779
  H   -1.259770066  -5.007497443  -3.580020485
  C   -1.862200057  -4.687419003  -1.471104666
  H   -2.199486547  -3.829024265  -0.830529215
  O   -2.832901515  -5.743732995  -1.225820250
  C   -2.196404399  -6.936044426  -0.747718283
  H   -2.877446131  -7.306685293   0.057662424
  C   -0.488076547  -5.214920649  -0.983942012
  H    0.261681051  -5.331786758  -1.802620956
  C   -0.795255417  -6.547745758  -0.274336929
  H   -0.021101969  -7.314645540  -0.489999565
  H   -0.787284124  -6.432858800   0.832069597
  O   -0.051706816  -4.192858573  -0.067418398
  O   -3.899099478  -1.722397064  -3.928479076
  O   -2.953334688  -1.884801772  -1.577005384
  N   -2.203598386  -7.946342895  -1.865274138
  C   -1.241870588  -7.911424642  -2.914586470
  N   -1.346853855  -8.922306185  -3.901545735
  C   -2.419189803  -9.871483674  -3.994220717
  C   -3.367946117  -9.830020002  -2.874097141
  C   -3.246857547  -8.903799572  -1.889515493
  O   -0.362702022  -7.068704843  -2.996459432
  H   -0.552035845  -8.986000323  -4.560543099
  O   -2.470096544 -10.574657043  -4.987854676
  C   -4.443495844 -10.859995760  -2.871568515
  H   -4.484540978 -11.415849981  -3.825509669
  H   -4.278222152 -11.615775925  -2.084761107
  H   -5.441717219 -10.426885399  -2.711060534
  H   -3.962028967  -8.856427256  -1.053952321
  P    1.446479390  -4.376608915   0.605981723
  O    1.906097854  -2.805058098   0.534294072
  C    3.140806343  -2.489461774  -0.144095006
  H    3.447474911  -1.566698168   0.400499212
  H    3.913587007  -3.276913728   0.004218850
  C    2.947558620  -2.203986090  -1.636345212
  H    3.532645153  -1.296875661  -1.936752080
  O    3.595589500  -3.315301045  -2.334807217
  C    2.883499166  -3.635558584  -3.542821496
  H    3.534408200  -3.308060656  -4.392236736
  C    1.510469054  -2.105779524  -2.189451482
  H    0.718150982  -2.401337572  -1.464613037
  C    1.520849414  -2.942954293  -3.475131318
  H    0.670022207  -3.645666303  -3.523257945
  H    1.392341221  -2.287003547  -4.372686199
  O    1.298195220  -0.714963779  -2.506245559
  O    2.296506620  -5.425928320   0.032101515
  O    1.069655835  -4.514418247   2.181052553
  N    2.760646343  -5.132140938  -3.615622618
  C    2.347369152  -5.697008298  -4.887076862
  N    2.261382978  -7.069149928  -5.018372607
  C    2.514264381  -7.881584845  -3.938589809
  C    2.957129481  -7.332219977  -2.689198223
  C    3.083276220  -5.972503122  -2.553692198
  O    2.086587219  -4.937234855  -5.820079277
  N    2.282872930  -9.219544495  -4.123150552
  H    2.587236698  -9.904214442  -3.447332875
  H    2.004116332  -9.570563169  -5.039436291
  H   -0.383353661  -1.838038869  -5.455925181
  H    3.441784866  -5.494320998  -1.616672825
  H    3.182254062  -7.979059763  -1.841888109
  H    0.444041384  -0.652263339  -2.997423792
  H   -5.246793806  -2.311639172  -5.123257697
  H   -3.864830876  -1.611876021  -1.288755530
  H    0.737780149  -3.700118261   2.657343037
  H    1.985785181  -7.233273262  -6.929527233
  H   -0.760413469   3.412912548  -6.878774293
