import onnx
import numpy as np
from onnx import helper, TensorProto
from onnx import numpy_helper

def create_simple_mt5_model():
    """Create a simple ONNX model that's guaranteed to work with MT5"""
    
    print("🔧 Creating simple MT5-compatible ONNX model...")
    
    # Define input - 19 features as expected
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [None, 19]
    )
    
    # Define output - single prediction value
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, [None, 1]
    )
    
    # Create simple weights matrix (19 inputs -> 1 output)
    # This will be a simple linear combination
    weights = np.random.randn(19, 1).astype(np.float32) * 0.1
    weights_tensor = numpy_helper.from_array(weights, name='weights')
    
    # Create bias
    bias = np.array([0.0], dtype=np.float32)
    bias_tensor = numpy_helper.from_array(bias, name='bias')
    
    # Create MatMul node
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['input', 'weights'],
        outputs=['matmul_out']
    )
    
    # Create Add node (add bias)
    add_node = helper.make_node(
        'Add',
        inputs=['matmul_out', 'bias'],
        outputs=['add_out']
    )
    
    # Create Sigmoid node to get probability between 0 and 1
    sigmoid_node = helper.make_node(
        'Sigmoid',
        inputs=['add_out'],
        outputs=['output']
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[matmul_node, add_node, sigmoid_node],
        name='SimpleModel',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weights_tensor, bias_tensor]
    )
    
    # Create the model with compatible IR version
    model = helper.make_model(graph, producer_name='MT5-Compatible')
    model.ir_version = 6  # Use IR version 6 for compatibility
    model.opset_import[0].version = 11  # Use opset 11 for better compatibility
    
    # Verify the model
    onnx.checker.check_model(model)
    
    return model

def create_classification_model():
    """Create a classification model with both class and probability outputs"""
    
    print("🔧 Creating classification MT5-compatible ONNX model...")
    
    # Define input
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, [None, 19]
    )
    
    # Define outputs
    class_output = helper.make_tensor_value_info(
        'class_output', TensorProto.INT64, [None]
    )
    
    prob_output = helper.make_tensor_value_info(
        'prob_output', TensorProto.FLOAT, [None, 2]
    )
    
    # Create weights for logistic regression
    weights = np.random.randn(19, 2).astype(np.float32) * 0.1
    weights_tensor = numpy_helper.from_array(weights, name='weights')
    
    bias = np.array([0.0, 0.0], dtype=np.float32)
    bias_tensor = numpy_helper.from_array(bias, name='bias')
    
    # MatMul
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['input', 'weights'],
        outputs=['matmul_out']
    )
    
    # Add bias
    add_node = helper.make_node(
        'Add',
        inputs=['matmul_out', 'bias'],
        outputs=['add_out']
    )
    
    # Softmax for probabilities
    softmax_node = helper.make_node(
        'Softmax',
        inputs=['add_out'],
        outputs=['prob_output'],
        axis=1
    )
    
    # ArgMax for class prediction
    argmax_node = helper.make_node(
        'ArgMax',
        inputs=['add_out'],
        outputs=['class_output'],
        axis=1
    )
    
    # Create the graph
    graph = helper.make_graph(
        nodes=[matmul_node, add_node, softmax_node, argmax_node],
        name='ClassificationModel',
        inputs=[input_tensor],
        outputs=[class_output, prob_output],
        initializer=[weights_tensor, bias_tensor]
    )
    
    # Create the model with compatible IR version
    model = helper.make_model(graph, producer_name='MT5-Compatible')
    model.ir_version = 6  # Use IR version 6 for compatibility
    model.opset_import[0].version = 11  # Use opset 11 for better compatibility
    
    # Verify the model
    onnx.checker.check_model(model)
    
    return model

if __name__ == "__main__":
    try:
        # Create simple regression model
        print("📦 Creating simple regression model...")
        simple_model = create_simple_mt5_model()
        onnx.save(simple_model, "simple_test_model.onnx")
        print("✅ Saved: simple_test_model.onnx")
        
        # Create classification model
        print("📦 Creating classification model...")
        class_model = create_classification_model()
        onnx.save(class_model, "classification_test_model.onnx")
        print("✅ Saved: classification_test_model.onnx")
        
        # Test both models
        import onnxruntime as ort
        
        # Test simple model
        print("\n🧪 Testing simple model...")
        session = ort.InferenceSession("simple_test_model.onnx")
        test_input = np.random.rand(1, 19).astype(np.float32)
        result = session.run(None, {'input': test_input})
        print(f"✅ Simple model output: {result[0][0][0]:.4f}")
        
        # Test classification model
        print("\n🧪 Testing classification model...")
        session = ort.InferenceSession("classification_test_model.onnx")
        result = session.run(None, {'input': test_input})
        print(f"✅ Class prediction: {result[0][0]}")
        print(f"✅ Probabilities: [{result[1][0][0]:.4f}, {result[1][0][1]:.4f}]")
        
        print("\n🎯 Both models created and tested successfully!")
        print("📁 Copy one of these to your MT5 Files directory for testing")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}") 