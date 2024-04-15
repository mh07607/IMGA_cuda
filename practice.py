import taichi as ti
import math

vec3 = ti.math.vec3
ti.init(arch=ti.cpu, default_fp=ti.f64)

# This example shows how we can imitate abstract classes in taichi
@ti.dataclass
class Sphere:
    center: vec3
    radius: ti.f32
    
    @ti.func
    def two(self):
        return 2
    
@ti.func
def area(self):
    # a function to run in taichi scope
    return 4 * math.pi * self.radius * self.radius

# Derived class with added method of area
MySphere = Sphere
MySphere.methods = {'area': area}
    
feild = MySphere.field(shape=(5,))
    

    
# Question: Can i use a type in ti func but define that type later?
# Hell yeah! stringify the type. Example below
@ti.dataclass
class ABC:
    @ti.func
    def two(self, a) -> "Lobster":
        return 2
Lobster = ti.i32



@ti.kernel
def test_sphere():
    feild[0] = MySphere(center=vec3(0, 0, 0), radius=1)
    print(feild[0].area())
    
    #using __getattr__ #not needed though
    print(feild[0].__getattribute__('area')())
    
    abc = ABC()
    print(abc.two(2))    
    
    



if __name__ == '__main__':
    test_sphere()
    