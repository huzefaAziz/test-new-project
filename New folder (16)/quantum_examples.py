"""
Practical Examples: Quantum Meta-Class Fusion Applications
Real-world use cases and advanced patterns
"""

from quantum_metaclass_fusion import (
    QuantumMetaClass, QuantumState, QuantumAttribute, 
    QuantumMixin, ClassFusionEngine
)
import random


# ============================================================================
# Example 1: Quantum Configuration System
# ============================================================================

class DevelopmentConfig(metaclass=QuantumMetaClass,
                       quantum_attrs={
                           'debug_mode': {True: 0.9, False: 0.1},
                           'log_level': {'DEBUG': 0.7, 'INFO': 0.3}
                       }):
    """Development configuration with quantum uncertainty"""
    
    database_url = "localhost:5432"
    api_timeout = 30
    
    def get_settings(self):
        return {
            'debug': self.debug_mode,
            'log_level': self.log_level,
            'database': self.database_url
        }


class ProductionConfig(metaclass=QuantumMetaClass,
                      quantum_attrs={
                          'debug_mode': {False: 1.0},
                          'log_level': {'ERROR': 0.6, 'WARNING': 0.4}
                      }):
    """Production configuration"""
    
    database_url = "prod-db.example.com:5432"
    api_timeout = 60
    cache_enabled = True
    
    def get_settings(self):
        return {
            'debug': self.debug_mode,
            'log_level': self.log_level,
            'database': self.database_url,
            'cache': self.cache_enabled
        }


# ============================================================================
# Example 2: Quantum State Machine
# ============================================================================

class TrafficLightState(metaclass=QuantumMetaClass,
                       quantum_attrs={
                           'color': {
                               'red': 0.33,
                               'yellow': 0.33,
                               'green': 0.34
                           }
                       }):
    """Traffic light with quantum superposition of states"""
    
    def __init__(self, intersection: str):
        self.intersection = intersection
        self.timer = 0
    
    def observe_state(self):
        """Collapse to definite state"""
        current_color = self.color
        print(f"Traffic light at {self.intersection}: {current_color}")
        return current_color
    
    def get_action(self):
        """Get action based on current state"""
        state = self.color
        actions = {
            'red': 'STOP',
            'yellow': 'CAUTION',
            'green': 'GO'
        }
        return actions.get(state, 'UNKNOWN')


# ============================================================================
# Example 3: Quantum AI Agent Behaviors
# ============================================================================

class ExplorationBehavior(metaclass=QuantumMetaClass,
                         quantum_attrs={
                             'strategy': {
                                 'random': 0.4,
                                 'greedy': 0.3,
                                 'systematic': 0.3
                             }
                         }):
    """AI agent exploration behavior"""
    
    def __init__(self, name: str):
        self.name = name
        self.exploration_rate = 0.8
    
    def decide_action(self, options):
        strategy = self.strategy
        if strategy == 'random':
            return random.choice(options)
        elif strategy == 'greedy':
            return options[0] if options else None
        else:
            return options[-1] if options else None


class LearningBehavior(metaclass=QuantumMetaClass,
                      quantum_attrs={
                          'learning_rate': {0.01: 0.3, 0.1: 0.5, 0.5: 0.2},
                          'method': {
                              'reinforcement': 0.5,
                              'supervised': 0.3,
                              'unsupervised': 0.2
                          }
                      }):
    """AI agent learning behavior"""
    
    def __init__(self, name: str):
        self.name = name
        self.knowledge = []
    
    def learn(self, experience):
        rate = self.learning_rate
        method = self.method
        print(f"{self.name} learning via {method} at rate {rate}")
        self.knowledge.append(experience)


# ============================================================================
# Example 4: Quantum Plugin System
# ============================================================================

class PluginBase(metaclass=QuantumMetaClass, fusion_enabled=True):
    """Base class for plugins"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
    
    def execute(self):
        raise NotImplementedError


class AuthenticationPlugin(PluginBase,
                          quantum_attrs={
                              'auth_method': {
                                  'oauth': 0.4,
                                  'jwt': 0.4,
                                  'basic': 0.2
                              }
                          }):
    """Authentication plugin"""
    
    def execute(self):
        return f"Authenticating via {self.auth_method}"


class CachingPlugin(PluginBase,
                   quantum_attrs={
                       'cache_strategy': {
                           'memory': 0.5,
                           'redis': 0.3,
                           'disk': 0.2
                       }
                   }):
    """Caching plugin"""
    
    def execute(self):
        return f"Caching with {self.cache_strategy}"


# ============================================================================
# Example 5: Quantum Testing Framework
# ============================================================================

class QuantumTestCase(QuantumMixin, metaclass=QuantumMetaClass):
    """Test case that explores multiple execution paths"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.results = []
    
    def run_quantum_test(self, test_func, iterations=10):
        """Run test multiple times to explore quantum states"""
        print(f"\n=== Running Quantum Test: {self.test_name} ===")
        outcomes = {}
        
        for i in range(iterations):
            result = test_func()
            outcome = str(result)
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        print(f"Test outcomes across {iterations} iterations:")
        for outcome, count in outcomes.items():
            probability = count / iterations
            print(f"  {outcome}: {probability:.1%} ({count}/{iterations})")
        
        return outcomes


# ============================================================================
# Practical Application Examples
# ============================================================================

def example_1_config_fusion():
    """Demonstrate configuration fusion for hybrid environments"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Configuration Fusion")
    print("="*70)
    
    # Fuse development and production configs for staging
    StagingConfig = DevelopmentConfig.fuse(
        ProductionConfig, 
        fusion_strategy='superpose'
    )
    
    staging = StagingConfig()
    print(f"\nStaging Configuration (Superposed):")
    print(f"  Debug mode: {staging.debug_mode}")
    print(f"  Log level: {staging.log_level}")
    print(f"  Database: {staging.database_url}")


def example_2_traffic_simulation():
    """Simulate quantum traffic light system"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Quantum Traffic Light Simulation")
    print("="*70)
    
    lights = [
        TrafficLightState("Main St & 1st Ave"),
        TrafficLightState("Main St & 2nd Ave"),
        TrafficLightState("Main St & 3rd Ave")
    ]
    
    print("\nObserving traffic lights:")
    for light in lights:
        action = light.get_action()
        state = light.color
        print(f"  {light.intersection}: {state} - {action}")


def example_3_ai_agent_fusion():
    """Create hybrid AI agent through fusion"""
    print("\n" + "="*70)
    print("EXAMPLE 3: AI Agent Behavior Fusion")
    print("="*70)
    
    # Fuse exploration and learning behaviors
    HybridAgent = ExplorationBehavior.fuse(
        LearningBehavior,
        fusion_strategy='merge'
    )
    
    agent = HybridAgent("QuantumBot")
    print(f"\nHybrid agent created: {agent.name}")
    print(f"  Exploration strategy: {agent.strategy}")
    print(f"  Learning rate: {agent.learning_rate}")
    print(f"  Learning method: {agent.method}")
    
    # Agent makes decisions
    options = ['action_A', 'action_B', 'action_C']
    action = agent.decide_action(options)
    print(f"\n  Agent chose: {action}")
    
    # Agent learns
    agent.learn(f"Executed {action}")


def example_4_plugin_composition():
    """Compose plugins through quantum fusion"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Quantum Plugin Composition")
    print("="*70)
    
    # Create individual plugins
    auth = AuthenticationPlugin("Auth")
    cache = CachingPlugin("Cache")
    
    print(f"\nIndividual plugins:")
    print(f"  {auth.execute()}")
    print(f"  {cache.execute()}")
    
    # Fuse plugins for combined functionality
    CompositePlugin = AuthenticationPlugin.fuse(
        CachingPlugin,
        fusion_strategy='merge'
    )
    
    composite = CompositePlugin("Composite")
    print(f"\nComposite plugin:")
    print(f"  Name: {composite.name}")
    print(f"  Auth: {composite.auth_method}")
    print(f"  Cache: {composite.cache_strategy}")


def example_5_quantum_testing():
    """Demonstrate quantum testing framework"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Quantum Testing Framework")
    print("="*70)
    
    # Create test case
    test = QuantumTestCase("Traffic Light Test")
    
    # Define test that uses quantum states
    def traffic_test():
        light = TrafficLightState("Test Intersection")
        return light.color
    
    # Run quantum test
    test.run_quantum_test(traffic_test, iterations=20)


def example_6_dynamic_quantum_attrs():
    """Demonstrate dynamic quantum attribute creation"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Dynamic Quantum Attributes")
    print("="*70)
    
    class DynamicQuantumClass(QuantumMixin, metaclass=QuantumMetaClass):
        def __init__(self):
            self.static_value = "Static"
    
    obj = DynamicQuantumClass()
    
    # Create quantum attributes at runtime
    print("\nCreating quantum attributes dynamically:")
    
    obj.create_superposition('priority', {
        'high': 0.5,
        'medium': 0.3,
        'low': 0.2
    })
    
    obj.create_superposition('status', {
        'active': 0.7,
        'pending': 0.2,
        'inactive': 0.1
    })
    
    print(f"  Priority superposition: {obj.get_superposition_state('priority')}")
    print(f"  Status superposition: {obj.get_superposition_state('status')}")
    
    print(f"\nObserving quantum attributes:")
    print(f"  Priority: {obj.observe_attribute('priority')}")
    print(f"  Status: {obj.observe_attribute('status')}")


def run_all_examples():
    """Run all practical examples"""
    print("\n" + "="*70)
    print("QUANTUM META-CLASS FUSION: PRACTICAL EXAMPLES")
    print("="*70)
    
    example_1_config_fusion()
    example_2_traffic_simulation()
    example_3_ai_agent_fusion()
    example_4_plugin_composition()
    example_5_quantum_testing()
    example_6_dynamic_quantum_attrs()
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_examples()
