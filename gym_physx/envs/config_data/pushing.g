floor 	{  shape:ssBox, size:[3, 3, 0.1, 0.02], contact:-1 X:<[0, 0, 0.5, 1, 0, 0, 0]> 
    fixed, contact, logical:{ }
    friction:1.}
box(floor) 	{  shape:ssBox, size:[0.4, 0.4, 0.2, 0.05], contact:-1 X:<[0, 0, 0.7, 1, 0, 0, 0]>
    mass:1
    joint:rigid
    friction:.1}
finger 	{  shape:sphere, size:[0.06], contact:-1 X:<[0.9, 0, 0.9, 1, 0, 0, 0]> 
    friction:.1}

