import onnx
import onnxruntime as ort
import numpy as np
import os

def verify_onnx_model(model_path):
    """Verify ONNX model compatibility with MetaTrader 5"""
    
    print(f"🔍 Verifying ONNX model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        # Load the model
        print("📥 Loading ONNX model...")
        model = onnx.load(model_path)
        
        # Check model version and opset
        print(f"📊 Model version: {model.model_version}")
        print(f"📋 Producer: {model.producer_name}")
        print(f"🔢 IR version: {model.ir_version}")
        
        # Check opset versions (MT5 typically supports up to opset 14)
        for opset in model.opset_import:
            print(f"🎯 Opset domain '{opset.domain}': version {opset.version}")
            if opset.version > 14:
                print(f"⚠️  Warning: Opset version {opset.version} might not be supported by MT5")
        
        # Check input/output specifications
        print("\n📥 Model Inputs:")
        for i, input_spec in enumerate(model.graph.input):
            print(f"  {i}: {input_spec.name}")
            shape = [dim.dim_value for dim in input_spec.type.tensor_type.shape.dim]
            print(f"     Shape: {shape}")
            print(f"     Type: {input_spec.type.tensor_type.elem_type}")
        
        print("\n📤 Model Outputs:")
        for i, output_spec in enumerate(model.graph.output):
            print(f"  {i}: {output_spec.name}")
            shape = [dim.dim_value for dim in output_spec.type.tensor_type.shape.dim]
            print(f"     Shape: {shape}")
            print(f"     Type: {output_spec.type.tensor_type.elem_type}")
        
        # Test with ONNX Runtime
        print("\n🔄 Testing with ONNX Runtime...")
        session = ort.InferenceSession(model_path)
        
        # Create test input (19 features as expected by your EA)
        test_input = np.random.rand(1, 19).astype(np.float32)
        print(f"🧪 Test input shape: {test_input.shape}")
        
        # Get input name
        input_name = session.get_inputs()[0].name
        print(f"📝 Input name: {input_name}")
        
        # Run prediction
        outputs = session.run(None, {input_name: test_input})
        
        print(f"✅ Model executed successfully!")
        print(f"📊 Number of outputs: {len(outputs)}")
        
        for i, output in enumerate(outputs):
            if hasattr(output, 'shape'):
                print(f"📊 Output {i} shape: {output.shape}")
                print(f"📈 Output {i} sample: {output.flatten()[:5] if hasattr(output, 'flatten') else output}")
            else:
                print(f"📊 Output {i} type: {type(output)}")
                print(f"📈 Output {i} content: {output}")
        
        # Check if output format matches MT5 expectations
        if len(outputs) >= 2:
            # Expected: [class_output, probability_output]
            class_output = outputs[0]  # output_label
            prob_output = outputs[1]   # output_probability
            
            print(f"✅ Model has separate class and probability outputs")
            print(f"📊 Class prediction: {class_output[0] if hasattr(class_output, '__len__') and len(class_output) > 0 else class_output}")
            
            if hasattr(prob_output, 'shape'):
                print(f"📊 Probability structure: {prob_output.shape}")
                if len(prob_output.shape) >= 2 and prob_output.shape[1] >= 2:
                    print(f"📊 Class 0 prob: {prob_output[0][0]:.4f}")
                    print(f"📊 Class 1 prob: {prob_output[0][1]:.4f}")
                else:
                    print("⚠️  Probability output structure may need adjustment for MT5")
            else:
                print(f"📊 Probability type: {type(prob_output)}")
                print(f"📊 Probability content: {prob_output}")
                
                # Handle scikit-learn probability output format
                if isinstance(prob_output, dict):
                    print("✅ Scikit-learn probability dictionary detected")
                    for key, value in prob_output.items():
                        print(f"   {key}: {value}")
                elif hasattr(prob_output, '__iter__'):
                    print("✅ Iterable probability output detected")
                    try:
                        prob_list = list(prob_output.flatten()) if hasattr(prob_output, 'flatten') else list(prob_output)
                        print(f"   Probabilities: {prob_list}")
                    except:
                        print(f"   Could not convert to list: {prob_output}")
        else:
            print("⚠️  Model should output both class and probabilities for MT5")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying model: {str(e)}")
        return False

def create_mt5_compatible_model(input_model_path, output_model_path):
    """Attempt to create a more MT5-compatible version"""
    
    print(f"\n🔧 Attempting to create MT5-compatible model...")
    
    try:
        # Load original model
        model = onnx.load(input_model_path)
        
        # Convert to lower opset if needed
        from onnx import version_converter
        
        # Try to convert to opset 14 (commonly supported by MT5)
        target_version = 14
        print(f"🔄 Converting to opset version {target_version}...")
        
        converted_model = version_converter.convert_version(model, target_version)
        
        # Save the converted model
        onnx.save(converted_model, output_model_path)
        print(f"✅ Converted model saved: {output_model_path}")
        
        # Verify the converted model
        return verify_onnx_model(output_model_path)
        
    except Exception as e:
        print(f"❌ Error converting model: {str(e)}")
        return False

if __name__ == "__main__":
    # Check current directory for ONNX files
    current_dir = os.getcwd()
    onnx_files = [f for f in os.listdir(current_dir) if f.endswith('.onnx')]
    
    if not onnx_files:
        print("❌ No ONNX files found in current directory")
        print("💡 Please make sure your ONNX model file is in the same directory as this script")
    else:
        print(f"📁 Found ONNX files: {onnx_files}")
        
        # Check the expected model file
        expected_file = "xauusd_optimized_model.onnx"
        if expected_file in onnx_files:
            print(f"\n🎯 Checking your model: {expected_file}")
            success = verify_onnx_model(expected_file)
            
            if not success:
                print(f"\n🔧 Attempting to create compatible version...")
                compatible_file = "xauusd_optimized_model_mt5.onnx"
                create_mt5_compatible_model(expected_file, compatible_file)
        else:
            print(f"\n⚠️  Expected file '{expected_file}' not found")
            print("🔍 Checking first available ONNX file...")
            verify_onnx_model(onnx_files[0]) 