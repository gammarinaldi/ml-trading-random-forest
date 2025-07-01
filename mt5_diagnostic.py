import os
import glob

def check_mt5_setup():
    """Comprehensive MT5 ONNX setup diagnostics"""
    
    print("🔍 MetaTrader 5 ONNX Diagnostics")
    print("=" * 50)
    
    # Check the MT5 directory
    mt5_base = r"C:\Users\gamma\AppData\Roaming\MetaQuotes\Terminal"
    print(f"📁 MT5 Base Directory: {mt5_base}")
    
    if not os.path.exists(mt5_base):
        print("❌ MT5 base directory not found!")
        return
    
    # Find terminal directories
    terminal_dirs = [d for d in os.listdir(mt5_base) if os.path.isdir(os.path.join(mt5_base, d))]
    print(f"📂 Terminal directories found: {len(terminal_dirs)}")
    
    for terminal_dir in terminal_dirs:
        print(f"   - {terminal_dir}")
        
        # Check MQL5/Files directory
        files_dir = os.path.join(mt5_base, terminal_dir, "MQL5", "Files")
        if os.path.exists(files_dir):
            print(f"   ✅ Files directory exists")
            
            # Check for ONNX files
            onnx_files = glob.glob(os.path.join(files_dir, "*.onnx"))
            print(f"   📊 ONNX files found: {len(onnx_files)}")
            
            for onnx_file in onnx_files:
                filename = os.path.basename(onnx_file)
                size = os.path.getsize(onnx_file)
                print(f"      - {filename} ({size} bytes)")
                
                # Check permissions
                try:
                    with open(onnx_file, 'rb') as f:
                        f.read(10)  # Try to read first 10 bytes
                    print(f"      ✅ File is readable")
                except Exception as e:
                    print(f"      ❌ File access error: {e}")
        else:
            print(f"   ❌ Files directory not found")
    
    print("\n🧪 Testing Current Directory ONNX Files")
    print("-" * 40)
    
    current_onnx = glob.glob("*.onnx")
    if current_onnx:
        for onnx_file in current_onnx:
            print(f"📄 Testing: {onnx_file}")
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(onnx_file)
                inputs = session.get_inputs()
                outputs = session.get_outputs()
                
                print(f"   ✅ Model loads successfully")
                print(f"   📥 Inputs: {len(inputs)}")
                for i, inp in enumerate(inputs):
                    print(f"      {i}: {inp.name} {inp.shape} {inp.type}")
                
                print(f"   📤 Outputs: {len(outputs)}")
                for i, out in enumerate(outputs):
                    print(f"      {i}: {out.name} {out.shape} {out.type}")
                    
            except Exception as e:
                print(f"   ❌ Model error: {e}")
    else:
        print("❌ No ONNX files found in current directory")
    
    print("\n💡 Recommendations:")
    print("-" * 40)
    print("1. ✅ Compile your updated EA (random_forest.mq5)")
    print("2. ✅ Attach EA to XAUUSD H1 chart")
    print("3. ✅ Check Experts tab for detailed logs")
    print("4. ✅ The EA now has fallback mode if ONNX fails")
    print("5. ✅ Test with simple_test_model.onnx first")

if __name__ == "__main__":
    check_mt5_setup() 