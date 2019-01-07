%mem=64gb
%nproc=28       
%Chk=snap_18.chk
# opt b3lyp/6-31g(d,p)  scf=(xqc, tight)  pop=min    

tetra snap_18 

2     1 
  O    2.620002673  -3.937001414  -7.795234307
  C    2.539243667  -2.567652605  -8.199569340
  H    3.447255125  -2.438919445  -8.826407361
  H    2.614352040  -1.918529155  -7.300195750
  C    1.262171554  -2.250776526  -8.980660591
  H    1.225899603  -1.178743672  -9.285788119
  O    1.281650911  -2.986386754 -10.257485645
  C    0.102697169  -3.759474126 -10.410470196
  H   -0.155221007  -3.745252720 -11.495229187
  C   -0.047933252  -2.690139534  -8.277120613
  H    0.162452924  -3.497160480  -7.528556941
  C   -0.928323503  -3.232276959  -9.414323502
  H   -1.546734897  -2.418908001  -9.852509761
  H   -1.662667122  -3.982666073  -9.060981971
  O   -0.656919410  -1.532361767  -7.704195481
  N    0.491550567  -5.189510073 -10.052987385
  C   -0.389192844  -6.279058340  -9.898812284
  C    0.382035627  -7.368258169  -9.405085246
  N    1.703971320  -6.917433797  -9.241270746
  C    1.757206834  -5.609675002  -9.615949252
  N   -1.727091419  -6.320940184 -10.194359500
  C   -2.329401975  -7.528075736  -9.952825390
  N   -1.644158097  -8.652563781  -9.444199242
  C   -0.217381870  -8.651692298  -9.118941363
  N   -3.688966768  -7.611388733 -10.165685081
  H   -4.159119851  -6.824250826 -10.611219289
  H   -4.165740832  -8.500680933 -10.226867642
  O    0.257195927  -9.664228962  -8.678506882
  H    2.637363630  -4.938352325  -9.569675155
  H   -2.142538565  -9.536728862  -9.263870272
  P   -0.880427176  -1.675185928  -6.052759763
  O   -1.527423547  -0.160133416  -5.894583813
  C   -0.776589906   0.720919420  -5.024723597
  H   -0.513885526   1.595108829  -5.660360490
  H    0.164552518   0.260016157  -4.645044349
  C   -1.705785131   1.135904725  -3.878572061
  H   -1.219945438   1.017208322  -2.875307997
  O   -1.885320862   2.585801886  -3.974277858
  C   -3.253960181   2.927200783  -4.124455495
  H   -3.412606768   3.808601597  -3.462983304
  C   -3.125177390   0.501983441  -3.930108883
  H   -3.320757838  -0.091466699  -4.858673328
  C   -4.091296474   1.691462731  -3.794637351
  H   -4.473444054   1.727373299  -2.744638755
  H   -4.997051228   1.572665178  -4.416455016
  O   -3.371883470  -0.333760872  -2.793941633
  O    0.389472492  -1.967168757  -5.360120845
  O   -2.089903211  -2.761679293  -6.142275938
  N   -3.434045611   3.357971774  -5.556784872
  C   -3.680802344   4.644224288  -6.030658906
  C   -3.673485181   4.566470230  -7.462490951
  N   -3.396061598   3.249783931  -7.859613006
  C   -3.242686558   2.546243964  -6.738462933
  N   -3.930714847   5.856526343  -5.384016048
  C   -4.163692020   6.973203219  -6.214796015
  N   -4.166123622   6.940604202  -7.534989582
  C   -3.917693365   5.742608315  -8.240328723
  H   -4.361769859   7.944667906  -5.706421163
  N   -3.924274062   5.784021506  -9.581833721
  H   -4.099438442   6.652546339 -10.088609355
  H   -3.752440113   4.943471529 -10.133758839
  H   -2.989615442   1.483531464  -6.670520405
  P   -2.398028147  -1.677636425  -2.647876594
  O   -1.411389375  -0.916696701  -1.590919005
  C   -0.199056177  -1.579096169  -1.155209503
  H    0.439229334  -1.841058125  -2.031853954
  H    0.300582357  -0.798171004  -0.546211136
  C   -0.652349662  -2.785694658  -0.338677616
  H   -1.224140644  -2.508573195   0.581367899
  O   -1.644873809  -3.411253744  -1.221890602
  C   -1.255647088  -4.782710259  -1.565543494
  H   -2.203557911  -5.357924693  -1.420867050
  C    0.420031184  -3.847328832  -0.027333717
  H    1.434518615  -3.577071248  -0.415597263
  C   -0.125748150  -5.159049650  -0.613476863
  H    0.698313147  -5.735842805  -1.107131617
  H   -0.487291901  -5.840802002   0.189127999
  O    0.427985916  -3.896259993   1.419275885
  O   -1.794883568  -2.099395060  -3.950474061
  O   -3.577426757  -2.647138945  -2.103573276
  N   -0.873814387  -4.790282029  -3.000700625
  C    0.424761000  -4.327125361  -3.407639573
  N    0.773157186  -4.561579344  -4.754180533
  C   -0.050368692  -5.238814775  -5.691239648
  C   -1.360057872  -5.670301602  -5.207189238
  C   -1.712504227  -5.474741393  -3.909462659
  O    1.185807627  -3.768534239  -2.638559882
  H    1.657780124  -4.130556218  -5.074406476
  O    0.395958120  -5.346167753  -6.832983403
  C   -2.262047779  -6.314572738  -6.203524004
  H   -1.684386126  -6.900554189  -6.936389869
  H   -2.980693715  -7.004056075  -5.738688804
  H   -2.827931058  -5.554276665  -6.766726316
  H   -2.668343760  -5.833573285  -3.504023511
  P    1.692325685  -4.673575419   2.120652722
  O    1.640883548  -6.059898255   1.229675167
  C    2.415301325  -7.182931261   1.715772247
  H    2.831466627  -7.008254838   2.727537106
  H    1.661764348  -7.998426351   1.785924567
  C    3.503556594  -7.546809702   0.696817979
  H    4.421697759  -7.918976114   1.210169164
  O    3.063229085  -8.718508044  -0.054152045
  C    2.548434456  -8.341404125  -1.343181534
  H    2.993639284  -9.072233568  -2.059081209
  C    3.819760819  -6.475977795  -0.380523218
  H    3.675470364  -5.421351882  -0.043201973
  C    2.928602233  -6.873897631  -1.570725240
  H    2.037057700  -6.203495831  -1.619091248
  H    3.432335550  -6.714488188  -2.542412913
  O    5.199317486  -6.489808712  -0.702715572
  O    1.713895616  -4.742590360   3.574720637
  O    2.972172524  -3.938205116   1.382435573
  N    1.057036988  -8.574237369  -1.290294745
  C    0.173841255  -7.962296031  -2.296241542
  N   -1.195316188  -8.068219041  -2.107681943
  C   -1.688685563  -8.833039157  -1.090405717
  C   -0.832459376  -9.529761804  -0.171902133
  C    0.525672492  -9.377224370  -0.294412564
  O    0.706349354  -7.355810203  -3.207888026
  N   -3.060051112  -8.856650866  -0.942618268
  H   -3.493710330  -9.517015901  -0.314936424
  H   -3.650593415  -8.489500527  -1.675260798
  H    1.851021902  -4.180130580  -7.203912173
  H    1.235053412  -9.881289334   0.385132413
  H   -1.249906982 -10.167548004   0.603532423
  H    5.497428176  -7.362957689  -1.043685576
  H   -2.363774667  -3.178749759  -5.241504240
  H   -3.576794514  -2.904012609  -1.126865594
  H    3.650299138  -3.481617727   1.947012960
  H    2.452914431  -7.482193380  -8.851337697
  H   -3.910843712   5.936709556  -4.364380565

