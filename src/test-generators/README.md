<H1>Test Generators</H1>
These scripts generate test data, whether that be images or raw edge points.

<h2>Point Generation</h2>
point-generator.py analytically computes edge points that can be submitted to Zernike-Moments or CRA.
Position can be either randomly generated or specified on execution.
Rotation can be randomly generated or specified on execution if position was specified

<h3>Usage</h3>
There are three ways to call point-generator.py:
<h4>Random position and rotation:</h4>
<code>python conicGenerator.py [num points] [noise; not implemented yet] [angle redundancy] [x resolution] [y resolution] [sensor width] [focal length]</code> <br/><br/>
<ul>
  <li>num points: the number of points to generate</li>
  <li>noise: how much noise error to add to the points (unimplemented)</li>
  <li>angle redundancy: the horizon appears more than this many radians away from the edge (may be removed soon)</li>
  <li>x resolution: the width of the image</li>
  <li>y resolution: the height of the image</li>
  <li>sensor width: the sensor width of the camera</li>
  <li>focal length: the focal length of the camera (may be changed to pixel size soon)</li>
</ul>
Example: <code>python conicGenerator.py 23 0 0.3 700 600 0.036 0.05</code>
<h4>Fixed position, random rotation</h4>
Same as random position and rotation, but the position is passed in first <br/>
<code>python conicGenerator.py [position.x] [position.y] [position.z] [... rest of the paramaters]</code>
<h4>Fixed position, fixed rotation</h4>
Same as random position and rotation, but position and rotation are passed in first <br/>
<code>python conicGenerator.py [position.x] [position.y] [position.z] [rotation.x] [rotation.y] [rotation.z] [... rest of the paramaters]</code><br/>
Rotation corresponds to a global XYZ rotation in radians (to be changed (like everything else))
