"""Photo-guided S7 PRO VERSION 2 surface reconstruction. Units: millimetres internally.
Requires: numpy, trimesh, manifold3d. Nominal dimensions are not measured CAD.
"""
import math,json,os
from pathlib import Path
import numpy as np
import trimesh
import manifold3d as mf

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'h10-v3';OUT.mkdir(exist_ok=True)
scene=trimesh.Scene();items=[]
def material(name,color,metal=.0,rough=.3):
 return trimesh.visual.material.PBRMaterial(name=name,baseColorFactor=color,metallicFactor=metal,roughnessFactor=rough,doubleSided=False)
WHITE=material('Pearl white satin housing',[226,229,231,255],.0,.38)
FRAME=material('Powder coated warm white support',[221,225,228,255],.045,.34)
BLACK=material('Charcoal front glass',[17,19,22,255],.025,.16)
DARK=material('Lens anodized black',[7,9,12,255],.48,.22)
HOLE=material('Deep recess',[5,7,9,255],.0,.63)
SILVER=material('Turned aluminium',[168,178,187,255],.88,.31)
BLUE=material('Blue coated optical glass',[18,45,75,255],.28,.085)
BLUE2=material('Inner optical coating',[31,76,122,255],.38,.11)
STEEL=material('Optical silver blue edge',[117,154,186,255],.88,.17)
SEAM=material('Shell seam',[112,118,123,255],.2,.46)
BRASS=material('Tripod brass socket',[173,136,59,255],.8,.3)

def rr(w,h,r,n=20):
 p=[]
 for cx,cy,start in [(w/2-r,h/2-r,0),(-w/2+r,h/2-r,90),(-w/2+r,-h/2+r,180),(w/2-r,-h/2+r,270)]:
  for t in np.linspace(start,start+90,n,endpoint=False):
   a=math.radians(t);p.append((cx+r*math.cos(a),cy+r*math.sin(a)))
 return np.array(p)

def prism(w,h,r,d,axis='z',pos=(0,0,0),angle=0):
 p=rr(w,h,r,12)
 if angle:
  a=math.radians(angle);p=p@np.array([[math.cos(a),math.sin(a)],[-math.sin(a),math.cos(a)]])
 m=mf.CrossSection([p]).extrude(d).translate((0,0,-d/2))
 if axis=='x':m=m.rotate((0,90,0))
 if axis=='y':m=m.rotate((-90,0,0))
 return m.translate(pos)

def rounded_prism(w,h,r,d,b=1.0):
 """Rounded XY outline with a real rolled edge on the front and back."""
 b=min(b,d/2-.0001);layers=[]
 for t in np.linspace(0,math.pi/2,6):
  inset=b*(1-math.sin(t));z=-d/2+b*(1-math.cos(t));layers.append((z,inset))
 layers += [(d/2-z*-1,inset) for z,inset in []]
 for t in np.linspace(math.pi/2,0,6):
  inset=b*(1-math.sin(t));z=d/2-b*(1-math.cos(t));layers.append((z,inset))
 v=[]
 for z,inset in layers:
  v.extend([(x,y,z) for x,y in rr(w-2*inset,h-2*inset,max(r-inset,.01))])
 N=len(v)//len(layers);f=[]
 for k in range(len(layers)-1):
  for i in range(N):
   a=k*N+i;c=k*N+(i+1)%N;f.extend([(a,c,c+N),(a,c+N,a+N)])
 v.extend([(0,0,-d/2),(0,0,d/2)])
 for i in range(N):
  j=(i+1)%N;f.extend([(len(v)-2,j,i),(len(v)-1,(len(layers)-1)*N+i,(len(layers)-1)*N+j)])
 mesh=trimesh.Trimesh(vertices=v,faces=f,process=True)
 mesh.fix_normals()
 return mf.Manifold(mf.Mesh(np.array(mesh.vertices,np.float32),np.array(mesh.faces,np.uint32)))

def add(name,m,mat,body=False):
 if isinstance(m,mf.Manifold):
  if m.is_empty():raise RuntimeError('Empty '+name)
  mm=m.to_mesh();mesh=trimesh.Trimesh(mm.vert_properties[:,:3],mm.tri_verts,process=True)
 else:mesh=m
 mesh.fix_normals()
 # Split caps and hard edges; blend only smoothly curved faces.
 mesh=trimesh.graph.smooth_shade(mesh,angle=math.radians(38))
 mesh.visual=trimesh.visual.TextureVisuals(material=mat)
 mesh.vertices/=1000
 transform=np.eye(4)
 if body:
  transform=trimesh.transformations.rotation_matrix(math.radians(-8),[1,0,0]);transform[:3,3]=[0,.183,0]
 scene.add_geometry(mesh,node_name=name,geom_name=name,transform=transform)
 items.append((name,mesh,transform))
 return mesh

def disc(rad,depth,pos=(0,0,0),axis='z',inner=0):
 m=mf.Manifold.cylinder(depth,rad,rad,128,True)
 if inner:m=m-mf.Manifold.cylinder(depth+2,inner,inner,128,True)
 if axis=='x':m=m.rotate((0,90,0))
 if axis=='y':m=m.rotate((-90,0,0))
 return m.translate(pos)

# Stand: thin front outline, deeper base, rounded continuous frame.
# Non-uniform border: 6 mm sides, 6 mm top, 20 mm base, matching image silhouette.
outer=mf.CrossSection([rr(316,330,31,28)])
inner=mf.CrossSection([rr(304,304,25,28)+[0,7]])
stand=(outer-inner).extrude(80).translate((0,165,-40))
# Depth tapers towards the top as visible in side references.
def taper(v):
 a=np.asarray(v);a[:,2]*=(1-.22*a[:,1]/330);return a
stand=stand.warp_batch(taper)
cuts=[]
for side in [-1,1]:
 clip=prism(55,180,4,16,axis='x',pos=(side*155,173,0))
 hub=disc(21,24,(side*155,183,0),'x')
 for y in np.arange(63,291,10.5):
  cutter=prism(88,3.3,1.65,22,'x',(side*155,float(y),0),angle=46)
  cutter=(cutter^clip)-hub
  if not cutter.is_empty():cuts.append(cutter)
stand=mf.Manifold.batch_boolean([stand]+cuts,mf.OpType.Subtract)
add('STAND / continuous frame with through slots',stand,FRAME)
# Thin edge trim follows original stand outlines, deliberately low contrast.
for z in [-39.9,39.9]:
 trim=(mf.CrossSection([rr(315.7,329.7,30.9,28)])-mf.CrossSection([rr(315.0,329.0,30.6,28)])).extrude(.25).translate((0,165,z))
 trim=trim.warp_batch(taper);add('Stand outer seam '+str(z),trim,SEAM)
for side in [-1,1]:
 add('Pivot axle '+str(side),disc(13,12,(side*150,183,0),'x'),DARK)
 add('Pivot aluminium cap '+str(side),disc(17.8,1.7,(side*159,183,0),'x'),SILVER)
 for rad in np.linspace(2,16.6,30):
  add('Pivot brushed ring '+str(side)+' '+str(rad),disc(float(rad),.09,(side*159.9,183,0),'x',float(rad)-.065),SILVER)

# Housing: rounded horizontal footprint, thin top and bottom rolled edges.
# Local body centre at origin; assembly tilted around actual side pivots.
housing=rounded_prism(302,249,14,104,2.4).rotate((-90,0,0))
inside=prism(296,243,11,98,'y')
housing=housing-inside
body_cuts=[]
# Dense staggered vertical capsules across both side faces.
for side in [-1,1]:
 for col,z in enumerate(np.arange(-104,107,6.8)):
  for row,y in enumerate(np.arange(-36,39,12.2)):
   length=8.0 if (col+row)%2 else 12.2
   # Rear cooling array has larger slots.
   if -92<z<-49 and -29<y<31:length=10.4
   body_cuts.append(prism(2.6,length,1.3,9,'x',(side*150,float(y),float(z))))
# Rear coordinates are mirrored from the front view: the connector island is
# on the viewer's left in a rear view, therefore it sits on model +X.
for col,x in enumerate(np.arange(-130,131,6.9)):
 for row,y in enumerate(np.arange(-38,39,12.0)):
  if -31<x<118 and 12<y<51:continue
  if -122<x<-98 and -39<y<-3:continue
  body_cuts.append(prism(2.8,8 if (col+row)%2 else 12,1.4,10,'z',(float(x),float(y),-124)))
# Underside cooling slots and recessed cover.
for x in [-115,-57,-21]:
 for z in np.arange(-81,84,8):body_cuts.append(prism(27,2.3,1.15,8,'y',(x,-51,float(z))))
housing=mf.Manifold.batch_boolean([housing]+body_cuts,mf.OpType.Subtract)
add('BODY / rounded shell and real ventilation holes',housing,WHITE,True)
# Interior shadows, set back from shell and not visible through front panel.
add('Internal equipment volume',prism(289,236,10,86,'y'),HOLE,True)
seam=(mf.CrossSection([rr(302.1,249.1,14.05)])-mf.CrossSection([rr(301.5,248.5,13.75)])).extrude(.28).translate((0,0,-.14)).rotate((-90,0,0)).translate((0,47,0))
add('Top lid seam',seam,SEAM,True)
# Front panel and recessed optical assembly, NOT a flat painted circle.
face=rounded_prism(250,86,17,1.0,.24).translate((0,0,124.9))
face=face-disc(37.8,8,(49,-1,125))
add('FRONT / dark inset bezel',face,BLACK,True)
add('Infrared window',rounded_prism(41,12,6,.65,.16).translate((-82,-1,125.7)),DARK,True)
add('Lens mounting surround',disc(38,2,(49,-1,125.1),'z',32.8),DARK,True)
add('Lens barrel bevel',mf.Manifold.cylinder(4,33.7,32.7,144,True).translate((49,-1,127.5))-disc(27.5,10,(49,-1,127.5)),DARK,True)
add('Lens satin outer lip',disc(32.8,.7,(49,-1,129.65),'z',31.8),SILVER,True)
add('Lens black lip',disc(32.5,.8,(49,-1,130),'z',28.1),DARK,True)
for i in range(96):
 t=i*math.tau/96;x=49+32.1*math.cos(t);y=-1+32.1*math.sin(t)
 add('Lens knurl %03d'%i,disc(.24,.25,(x,y,130.5)),SEAM,True)
# Curved optical surface with recessed dark pupil and many thin lens edges.
def dome(radius,depth,z,mat,name):
 ringcount=20;N=128;v=[(49,-1,z+depth)];faces=[]
 for k in range(1,ringcount+1):
  r=radius*k/ringcount;zz=z+depth*(1-(r/radius)**2)
  for a in np.linspace(0,math.tau,N,endpoint=False):v.append((49+r*math.cos(a),-1+r*math.sin(a),zz))
 for i in range(N):faces.append((0,1+i,1+(i+1)%N))
 for k in range(ringcount-1):
  a=1+k*N;b=a+N
  for i in range(N):j=(i+1)%N;faces.extend([(a+i,b+i,b+j),(a+i,b+j,a+j)])
 add(name,trimesh.Trimesh(v,faces,process=False),mat,True)
dome(27.8,6.4,126.4,BLUE,'OPTICS / convex blue coated glass')
# Four subtle internal rings preserve the coated-lens look without making the
# optical surface read as a flat target.
for rad in [11,16,21,25.5]:
 z=126.4+6.4*(1-(rad/27.8)**2)+.018
 add('Optical coating edge '+str(rad),disc(rad,.035,(49,-1,z),'z',rad-.10),BLUE2,True)
dome(8.2,1.5,131.15,DARK,'Lens pupil')
# Rear connector island is left of centre, as shown in rear reference.
add('REAR / port surround',rounded_prism(140,35,4,1.6,.4).translate((48,29,-125)),DARK,True)
add('Rear power button',disc(6.2,1.0,(84,30,-126.4)),BLACK,True)
add('HDMI socket surround',prism(24,8,1.5,1.3,'z',(62,34,-126.5)),SEAM,True)
add('HDMI opening',prism(21,5.4,1.2,1.4,'z',(62,34,-127.2)),HOLE,True)
add('HDMI tongue',prism(17,.8,.25,1.6,'z',(62,34,-127.8)),DARK,True)
add('Audio socket rim',disc(4.2,1,(35,32,-126.3)),SEAM,True)
add('Audio socket hole',disc(2.6,1,(35,32,-127)),HOLE,True)
add('USB surround',prism(22,10,1,1.1,'z',(11,34,-126.2)),SEAM,True)
add('USB opening',prism(18,6.8,.5,1.2,'z',(11,34,-127)),HOLE,True)
add('USB tongue',prism(16,2,.3,1.3,'z',(11,33,-127.7)),DARK,True)
add('Rear indicator',disc(1.4,1,(-6,24,-126.9)),HOLE,True)
add('AC figure 8 surround',prism(15,25,7.3,1.7,'z',(-110,-22,-125)),DARK,True)
for y in [-27,-17]:
 add('AC opening '+str(y),disc(4,2,(-110,y,-126)),HOLE,True)
 add('AC contact '+str(y),disc(1.1,1.9,(-110,y,-127)),SILVER,True)
# Bottom screws, hatch, tripod insert and anti-slip rails.
for x in [-132,132]:
 for z in [-106,106]:
  add('Bottom screw',disc(2,1,(x,-52.2,z),'y'),DARK,True)
add('Service hatch seam',prism(72,67,5,.6,'y',(70,-52,0)),SEAM,True)
add('Service hatch',prism(71,66,4.5,.7,'y',(70,-52.1,0)),WHITE,True)
add('Stand tripod seat',disc(10,.6,(0,20.4,0),'y'),WHITE)
add('Tripod underside insert',disc(6,.8,(0,0,0),'y',3),BRASS)
for x in [-119,119]:add('Non slip rail',prism(9,57,4,1.2,'y',(x,-.3,0)),DARK)

scene.metadata={'description':'S7 PRO VERSION 2 photo-guided reconstruction v3. Realistic PBR material pass. Nominal height 330 mm, not measured CAD. Real ventilation holes. Body pivot tilt -8 degrees.'}
scene.export(OUT/'s7-pro-version-2.glb')
scene.export(ROOT/'s7-pro-version-2.glb')
scene.export(OUT/'h10-projector-v3.glb')
scene.export(ROOT/'h10-projector-v3.glb')
# Save geometry for deterministic renderer and QA.
import pickle
with open(OUT/'scene.pkl','wb') as f:pickle.dump(items,f)
loaded=trimesh.load(ROOT/'s7-pro-version-2.glb',force='scene')
assert len(loaded.geometry)==len(items)
assert np.isfinite(loaded.bounds).all()
print('Exported',len(items),'objects;',sum(len(m.faces) for _,m,_ in items),'triangles; bounds',loaded.bounds.tolist(),flush=True)
