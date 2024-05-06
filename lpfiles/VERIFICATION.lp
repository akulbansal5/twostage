\ENCODING=ISO-8859-1
\Problem name: 

Maximize
 obj1: 2 y_1 + 3 y_2 + 3 y_3 + 4 y_4 - R + x_1
Subject To
 c_2: y0 + 16 y_1 + 19 y_2 + 23 y_3 + 28 y_4 - 100 R  = 0
 c_3: 2 y_1 + 3 y_2 + 4 y_3 + 5 y_4 - R + x_1 <= 5
 c_4: 6 y_1 + y_2 + 3 y_3 + 2 y_4 - R + x_2 <= 2
 c_5: y_1 <= 1
 c_6: y_2 <= 1
 c_7: y_3 <= 1
 c_8: y_4 <= 1
Bounds
      y0 Free
 0 <= y_1 <= 1
 0 <= y_2 <= 1
 0 <= y_3 <= 1
 0 <= y_4 <= 1
      R >= 0
 0 <= x_1 <= 1
 0 <= x_2 <= 1
Binaries
 y_1  y_2  y_3  y_4  x_1  x_2 
Generals
 y0  R 
End
