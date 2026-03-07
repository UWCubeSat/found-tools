<H1>Test Generators</H1>
These scripts generate test data, whether that be images or raw edge points.

<h2>Point Generation</h2>
point-generator.py analytically computes edge points that can be submitted to Zernike-Moments or CRA.
Position can be either randomly generated or specified on execution.
Rotation can be randomly generated or specified on execution if position was specified

<h3>Usage</h3>
There are three ways to call point-generator.py:
<h4>Random position and rotation:</h4>
<code>python point-generator.py [num points] [noise; not implemented yet] [angle redundancy] [x resolution] [y resolution] [sensor width] [focal length]</code> <br/><br/>
<ul>
  <li>num points: the number of points to generate</li>
  <li>noise: how much noise error to add to the points (unimplemented)</li>
  <li>angle redundancy: the horizon appears more than this many radians away from the edge (may be removed soon)</li>
  <li>x resolution: the width of the image</li>
  <li>y resolution: the height of the image</li>
  <li>sensor width: the sensor width of the camera</li>
  <li>focal length: the focal length of the camera (may be changed to pixel size soon)</li>
</ul>
Example: <code>python point-generator.py 23 0 0.3 700 600 0.036 0.05</code>
<h4>Fixed position, random rotation</h4>
Same as random position and rotation, but the position is passed in first <br/>
<code>python point-generator.py [position.x] [position.y] [position.z] [... rest of the paramaters]</code>
<h4>Fixed position, fixed rotation</h4>
Same as random position and rotation, but position and rotation are passed in first <br/>
<code>python point-generator.py [position.x] [position.y] [position.z] [rotation.x] [rotation.y] [rotation.z] [... rest of the paramaters]</code><br/>
Rotation corresponds to a global XYZ rotation in radians (to be changed (like everything else))

<h3>Output</h3>
The results are appended to pointOutput.txt in the following format:<br\>

```
POINTS FOR
local (rc) [-6.65458800358326, -0.1779251688783572, -0.7354764868268421]
global (rp) [0.7167086525860719, 6.438189385508616, 1.7006383123587012] 
with rotation [-56.743072836274024, 164.2314185171583, 72.50350056743514]:
{static_cast<decimal>(563.8804701214169), static_cast<decimal>(66.7912733800368)},{static_cast<decimal>(559.4548008809911), static_cast<decimal>(112.58215587790039)},{static_cast<decimal>(555.6378930744293), static_cast<decimal>(159.09540388081345)},{static_cast<decimal>(553.6248941177013), static_cast<decimal>(187.88767868685713)},{static_cast<decimal>(553.788625492359), static_cast<decimal>(587.4287948302749)},{static_cast<decimal>(559.8946581416474), static_cast<decimal>(107.70809728855897)},{static_cast<decimal>(556.1332398670804), static_cast<decimal>(152.54736814256754)},{static_cast<decimal>(556.2650699061115), static_cast<decimal>(150.83487825459935)},{static_cast<decimal>(563.9773089136995), static_cast<decimal>(65.85923507236154)},{static_cast<decimal>(564.4447406943715), static_cast<decimal>(61.39593825283362)},{static_cast<decimal>(570.2646980759681), static_cast<decimal>(9.976812850743437)},{static_cast<decimal>(557.2612534993655), static_cast<decimal>(138.26932993985065)},{static_cast<decimal>(559.0600786953607), static_cast<decimal>(117.02867341902697)},{static_cast<decimal>(569.9389614875669), static_cast<decimal>(12.680874331890895)},{static_cast<decimal>(557.0837868873962), static_cast<decimal>(140.46176532893597)},{static_cast<decimal>(553.602720545366), static_cast<decimal>(188.22793973755353)},{static_cast<decimal>(553.0358544492206), static_cast<decimal>(197.13355570161164)},{static_cast<decimal>(553.867672460827), static_cast<decimal>(184.19943599383095)},{static_cast<decimal>(556.271433287473), static_cast<decimal>(150.75252661623324)},{static_cast<decimal>(555.4003382700106), static_cast<decimal>(162.30252927723555)},{static_cast<decimal>(559.487284497024), static_cast<decimal>(112.21934025705433)},{static_cast<decimal>(565.0126311322574), static_cast<decimal>(56.05004985913748)},{static_cast<decimal>(562.4685656287348), static_cast<decimal>(80.685927347834)},
TPC:
{-0.07260182653404851, -0.9861641641565356, -0.14902756831325298,
0.8492207415743102, 0.01723376352152481, -0.5277566953387355,
-0.5230230462072379, 0.1648734021251624, -0.8362198600893045,
}

Calibrated conic equation: 1.0x^2 + 0.028843656609156755*2xy + -0.10596510642526717y^2 + -236.5793680872786*2x + 24.972778168972972*2y+-56192.05968037166 = 0

Other settings:
num points: 23
point noise: 0.0
angle redundancy: 0.3
resolution: 700.0x600.0
sensor width: 0.036
focal length: 0.05
FOV: 0.6911111611634242
AOR: [-0.14902757 -0.5277567  -0.83621986]
```
The line starting with <code>{static_cast<decimal>(563.8804701214169),</code> can be copy-pasted straight into C++, i.e <code>Points pts = {{static_cast<decimal>(117), ...</code>
The calibrated conic equation can be copy-pasted into desmos, but any scientific notation will have to be fixed manually.
TPC is the global to local coordinate transformation matrix.
AOR is the Earth's axis of rotation.
