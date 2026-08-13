"""
CREATINE AI: Polymorphic OOP Implementation
No algebra, pure object-oriented design with visual output
"""

from abc import ABC, abstractmethod
from PIL import Image, ImageDraw, ImageFont
import random
import os

# ============ BASE ABSTRACT CLASS ============
class CreatineMolecule(ABC):
    """Abstract base class for creatine molecular structures"""
    
    def __init__(self, name, energy_level):
        self.name = name
        self.energy_level = energy_level
        self.stability = 100
        self._hidden_phosphate = 3  # Encapsulated attribute
    
    @abstractmethod
    def structure_pattern(self):
        """Each creatine type has unique molecular pattern"""
        pass
    
    @abstractmethod
    def color_scheme(self):
        """Visual color representation"""
        pass
    
    def energize(self, amount):
        """Polymorphic method - behavior varies by subclass"""
        self.energy_level = min(100, self.energy_level + amount)
        self.stability -= amount * 0.5
        return f"⚡ {self.name} energized by {amount}!"
    
    def display_info(self):
        return f"{self.name} | Energy: {self.energy_level}% | Stability: {self.stability}%"
    
    def create_molecular_image(self, size=(400, 400)):
        """Generate visual representation (IMAGE YES)"""
        img = Image.new('RGB', size, color='black')
        draw = ImageDraw.Draw(img)
        
        # Draw molecular structure based on pattern
        pattern = self.structure_pattern()
        colors = self.color_scheme()
        
        # Draw atoms (circles) with polymorphism
        for i, (x, y) in enumerate(pattern):
            color = colors[i % len(colors)]
            draw.ellipse([x-20, y-20, x+20, y+20], fill=color, outline='white', width=2)
            
            # Label atoms
            labels = ['C', 'N', 'O', 'P', 'H']
            draw.text((x-5, y-8), labels[i % len(labels)], fill='white')
        
        # Draw bonds (lines)
        for i in range(len(pattern)-1):
            draw.line([pattern[i], pattern[i+1]], fill='lime', width=3)
        
        # Title
        draw.text((50, 20), f"🧬 {self.name}", fill='cyan', font=None)
        draw.text((50, 50), f"Energy: {self.energy_level}%", fill='yellow')
        
        return img

# ============ CONCRETE SUBCLASSES ============
class PureCreatine(CreatineMolecule):
    """Most stable form - baseline"""
    
    def structure_pattern(self):
        # Hexagonal arrangement
        return [(200, 100), (150, 170), (100, 250), 
                (200, 300), (300, 250), (250, 170)]
    
    def color_scheme(self):
        return ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    def energize(self, amount):
        # Override: Pure form gains energy efficiently
        self.energy_level = min(100, self.energy_level + amount * 1.2)
        self.stability -= amount * 0.3
        return f"💪 {self.name} PURE POWER! +{amount*1.2}!"

class HydrolyzedCreatine(CreatineMolecule):
    """Water-activated form - less stable but faster"""
    
    def structure_pattern(self):
        # Spiral water-like pattern
        return [(200, 50), (120, 120), (80, 200), 
                (120, 280), (200, 330), (280, 280)]
    
    def color_scheme(self):
        return ['#74B9FF', '#0984E3', '#6C5CE7', '#A29BFE', '#81ECEC']
    
    def energize(self, amount):
        # Hydrolyzed: unstable but quick boost
        self.energy_level = min(100, self.energy_level + amount * 1.5)
        self.stability -= amount * 0.8
        if self.stability < 30:
            return f"💧 {self.name} is degrading rapidly!"
        return f"🌊 {self.name} boosted by {amount*1.5}!"

class Phosphocreatine(CreatineMolecule):
    """High-energy phosphate carrier - most powerful"""
    
    def __init__(self, name, energy_level):
        super().__init__(name, energy_level)
        self._hidden_phosphate = 5  # Encapsulated extra phosphate
    
    def structure_pattern(self):
        # Star/burst pattern - phosphate rich
        return [(200, 50), (350, 150), (300, 320), 
                (100, 320), (50, 150), (200, 200)]
    
    def color_scheme(self):
        return ['#FF6B6B', '#FECA57', '#FF9F43', '#EE5A24', '#FDA7DF']
    
    def energize(self, amount):
        # Phosphocreatine: massive energy release
        self.energy_level = min(100, self.energy_level + amount * 2.0)
        self.stability -= amount * 0.2
        # Special phosphate release
        phosphate_used = min(self._hidden_phosphate, amount // 20)
        self._hidden_phosphate -= phosphate_used
        return f"⚗️ {self.name} releases {phosphate_used} phosphates! BOOSTED x2!"

# ============ FACTORY PATTERN ============
class CreatineFactory:
    """Factory to create polymorphic creatine objects"""
    
    @staticmethod
    def create_creatine(type_name, energy_level=50):
        types = {
            'pure': PureCreatine,
            'hydrolyzed': HydrolyzedCreatine,
            'phospho': Phosphocreatine
        }
        molecule_class = types.get(type_name.lower(), PureCreatine)
        return molecule_class(f"{type_name.capitalize()}Creatine", energy_level)

# ============ COMPOSITE PATTERN ============
class CreatineComplex:
    """Composite structure combining multiple creatine types"""
    
    def __init__(self, name):
        self.name = name
        self.components = []
    
    def add(self, molecule):
        self.components.append(molecule)
    
    def total_energy(self):
        return sum(m.energy_level for m in self.components) / len(self.components)
    
    def display_all(self):
        return "\n".join([m.display_info() for m in self.components])
    
    def create_composite_image(self):
        """Combine all molecules into one image"""
        if not self.components:
            return Image.new('RGB', (800, 600), 'black')
        
        # Create larger canvas
        img = Image.new('RGB', (800, 600), color='#1a1a2e')
        draw = ImageDraw.Draw(img)
        
        # Draw each molecule in grid
        positions = [(0, 0), (400, 0), (0, 300), (400, 300)]
        for i, mol in enumerate(self.components[:4]):
            mol_img = mol.create_molecular_image((300, 300))
            img.paste(mol_img, positions[i])
            
            # Label
            draw.text((positions[i][0]+20, positions[i][1]+10), 
                     f"#{i+1}: {mol.name}", fill='white')
        
        # Composite stats
        draw.text((300, 550), f"🧪 COMPOSITE: {self.name}", fill='lime', font=None)
        draw.text((300, 570), f"Avg Energy: {self.total_energy():.1f}%", fill='yellow')
        
        return img

# ============ MAIN EXECUTION ============
if __name__ == "__main__":
    print("🧪 CREATINE AI - POLYMORPHIC OOP DEMO")
    print("="*50)
    
    # 1. Create polymorphic molecules using factory
    factory = CreatineFactory()
    molecules = [
        factory.create_creatine('pure', 60),
        factory.create_creatine('hydrolyzed', 45),
        factory.create_creatine('phospho', 70)
    ]
    
    # 2. Demonstrate polymorphism - same method, different behavior
    print("\n🔬 POLYMORPHISM IN ACTION:")
    for mol in molecules:
        print(mol.display_info())
        # Polymorphic energize method
        print(mol.energize(random.randint(10, 30)))
        print("-" * 30)
    
    # 3. Create composite structure
    print("\n🧩 CREATING COMPOSITE STRUCTURE...")
    complex_mol = CreatineComplex("MegaCreatine Complex")
    for mol in molecules:
        complex_mol.add(mol)
    
    print(complex_mol.display_all())
    print(f"Composite average energy: {complex_mol.total_energy():.1f}%")
    
    # 4. GENERATE IMAGES (YES IMAGE)
    print("\n🎨 GENERATING MOLECULAR IMAGES...")
    
    # Create output directory
    os.makedirs("creatine_output", exist_ok=True)
    
    # Generate individual molecule images
    for i, mol in enumerate(molecules):
        img = mol.create_molecular_image()
        img.save(f"creatine_output/molecule_{i}_{mol.name}.png")
        print(f"✅ Saved: molecule_{i}_{mol.name}.png")
    
    # Generate composite image
    composite_img = complex_mol.create_composite_image()
    composite_img.save("creatine_output/composite_complex.png")
    print("✅ Saved: composite_complex.png")
    
    # 5. ENCAPSULATION DEMO
    print("\n🔒 ENCAPSULATION DEMO:")
    for mol in molecules:
        # Can't access _hidden_phosphate directly
        try:
            print(f"{mol.name} hidden phosphate: {mol._hidden_phosphate}")
        except AttributeError:
            print(f"{mol.name} has protected phosphate data")
    
    print("\n" + "="*50)
    print("✅ DEMO COMPLETE! Check 'creatine_output' folder for images.")
    print("📁 Images generated using polymorphism + OOP patterns!")