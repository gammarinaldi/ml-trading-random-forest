#include <Trade/Trade.mqh>
#define   ModelName          "XAUUSDRandomForest"
#define   ONNXFilename       "simple_test_model.onnx"

input double lotsize = 0.1;          // Trade lot size
input double confidence_threshold = 0.55; // Minimum confidence to trade
input double tp_multiplier = 1.5;    // Take profit ATR multiplier
input double sl_multiplier = 1.0;    // Stop loss ATR multiplier
input int atr_period = 14;           // ATR period for dynamic TP/SL
input int max_positions = 1;         // Maximum concurrent positions
input bool use_fallback_mode = false; // Use simple prediction if ONNX fails

CTrade m_trade;
long model_handle = INVALID_HANDLE;
bool model_loaded = false;

//+------------------------------------------------------------------+
//| Calculate Close Ratio for trend analysis                         |
//+------------------------------------------------------------------+
double CalculateCloseRatio(int period)
{
    if(period <= 0) return 0.0;
    
    double current_close = iClose(_Symbol, PERIOD_H1, 1);
    double past_close = iClose(_Symbol, PERIOD_H1, period + 1);
    
    if(past_close == 0.0) return 0.0;
    return current_close / past_close;
}

//+------------------------------------------------------------------+
//| Calculate Trend for period                                       |
//+------------------------------------------------------------------+
double CalculateTrend(int period)
{
    if(period <= 1) return 0.0;
    
    double sum = 0.0;
    for(int i = 1; i <= period; i++)
    {
        double current = iClose(_Symbol, PERIOD_H1, i);
        double previous = iClose(_Symbol, PERIOD_H1, i + 1);
        if(previous != 0.0)
            sum += (current - previous) / previous;
    }
    return sum / period;
}

//+------------------------------------------------------------------+
//| Prepare 19 features for ML model (EXACT ORDER!)                 |
//+------------------------------------------------------------------+
bool PrepareFeatures(double &features[])
{
    ArrayResize(features, 19);
    
    // Get current bar data
    double open = iOpen(_Symbol, PERIOD_H1, 1);
    double high = iHigh(_Symbol, PERIOD_H1, 1);
    double low = iLow(_Symbol, PERIOD_H1, 1);
    double close = iClose(_Symbol, PERIOD_H1, 1);
    long tick_volume = iTickVolume(_Symbol, PERIOD_H1, 1);
    
    if(open == 0 || high == 0 || low == 0 || close == 0) return false;
    
    // Features 0-4: Basic OHLC + Volume
    features[0] = close;                                    // <CLOSE>
    features[1] = (double)tick_volume;                      // <TICKVOL>
    features[2] = open;                                     // <OPEN>
    features[3] = high;                                     // <HIGH>
    features[4] = low;                                      // <LOW>
    
    // Features 5-8: Candle Pattern Features
    features[5] = MathAbs(close - open);                    // BODY_SIZE
    features[6] = high - MathMax(open, close);              // UPPER_WICK
    features[7] = MathMin(open, close) - low;               // LOWER_WICK
    double price_range = high - low;
    features[8] = (price_range > 0) ? (close - low) / price_range : 0.5; // CLOSE_POSITION
    
    // Features 9-18: Trend Analysis (Close_Ratio & Trend)
    features[9] = CalculateCloseRatio(2);                   // Close_Ratio_2
    features[10] = CalculateTrend(2);                       // Trend_2
    features[11] = CalculateCloseRatio(5);                  // Close_Ratio_5
    features[12] = CalculateTrend(5);                       // Trend_5
    features[13] = CalculateCloseRatio(55);                 // Close_Ratio_55
    features[14] = CalculateTrend(55);                      // Trend_55
    features[15] = CalculateCloseRatio(125);                // Close_Ratio_125
    features[16] = CalculateTrend(125);                     // Trend_125
    features[17] = CalculateCloseRatio(750);                // Close_Ratio_750
    features[18] = CalculateTrend(750);                     // Trend_750
    
    return true;
}

//+------------------------------------------------------------------+
//| Simple fallback prediction using basic TA                       |
//+------------------------------------------------------------------+
bool GetFallbackPrediction(long &prediction, double &confidence)
{
    // Simple momentum-based prediction as fallback
    double current_close = iClose(_Symbol, PERIOD_H1, 1);
    double prev_close = iClose(_Symbol, PERIOD_H1, 2);
    double trend_5 = CalculateTrend(5);
    double trend_55 = CalculateTrend(55);
    
    if(current_close == 0 || prev_close == 0) return false;
    
    // Simple logic: if short and long term trends align
    bool bullish = (current_close > prev_close) && (trend_5 > 0.0001) && (trend_55 > 0.0001);
    bool bearish = (current_close < prev_close) && (trend_5 < -0.0001) && (trend_55 < -0.0001);
    
    if(bullish)
    {
        prediction = 1;
        confidence = 0.6; // Moderate confidence for fallback
        return true;
    }
    else if(bearish)
    {
        prediction = 0;
        confidence = 0.6; // Moderate confidence for fallback
        return true;
    }
    
    // No clear signal
    prediction = 0;
    confidence = 0.45; // Below threshold
    return true;
}

//+------------------------------------------------------------------+
//| Run ML model prediction                                          |
//+------------------------------------------------------------------+
bool GetMLPrediction(long &prediction, double &confidence)
{
    // Use fallback mode if model not loaded or explicitly enabled
    if(!model_loaded || use_fallback_mode)
    {
        Print("🔄 Using fallback prediction mode");
        return GetFallbackPrediction(prediction, confidence);
    }
    
    double features[19];
    if(!PrepareFeatures(features)) 
    {
        Print("⚠️ Feature preparation failed, using fallback");
        return GetFallbackPrediction(prediction, confidence);
    }
    
    // Prepare input for ONNX (simple test model expects 'input' tensor)
    matrix input_matrix(1, 19);
    for(int i = 0; i < 19; i++)
        input_matrix[0][i] = features[i];
    
    matrix output_matrix;
    
    // Run prediction
    bool success = OnnxRun(model_handle, ONNX_NO_CONVERSION, input_matrix, output_matrix);
    if(!success || output_matrix.Rows() == 0) 
    {
        Print("⚠️ ONNX prediction failed, using fallback");
        return GetFallbackPrediction(prediction, confidence);
    }
    
    // Handle different model output formats
    if(output_matrix.Cols() == 1)
    {
        // Simple model: single sigmoid output (0.0 to 1.0)
        double prob = output_matrix[0][0];
        prediction = (prob > 0.5) ? 1 : 0;
        confidence = (prediction == 1) ? prob : (1.0 - prob);
        Print("🤖 Simple Model - Probability: ", prob, " | Prediction: ", prediction, " | Confidence: ", confidence);
    }
    else if(output_matrix.Rows() >= 1)
    {
        // Scikit-learn model: try to get class prediction from first output
        prediction = (long)output_matrix[0][0];
        
        // For confidence, try second output if available, otherwise use default
        if(output_matrix.Cols() >= 2)
        {
            confidence = MathAbs(output_matrix[0][1]);
            if(confidence < 0.5) confidence = 0.6; // Minimum confidence
            if(confidence > 1.0) confidence = 0.9; // Maximum confidence
        }
        else
        {
            confidence = (prediction == 1) ? 0.65 : 0.60;
        }
        
        Print("🤖 Scikit Model - Raw output[0][0]: ", output_matrix[0][0], " | Prediction: ", prediction, " | Confidence: ", confidence);
    }
    else
    {
        Print("⚠️ Unexpected output format, using fallback");
        return GetFallbackPrediction(prediction, confidence);
    }
        
    return true;
}

//+------------------------------------------------------------------+
//| Check if new hour started                                        |
//+------------------------------------------------------------------+
bool NewHour()
{
    static datetime last_hour = 0;
    datetime current_hour = iTime(_Symbol, PERIOD_H1, 0);
    
    if(current_hour != last_hour)
    {
        last_hour = current_hour;
        return true;
    }
    return false;
}

//+------------------------------------------------------------------+
//| Check if positions exist                                         |
//+------------------------------------------------------------------+
bool HasOpenPositions()
{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {
        if(PositionGetString(POSITION_SYMBOL) == _Symbol)
            return true;
    }
    return false;
}

//+------------------------------------------------------------------+
//| Open Buy Trade                                                   |
//+------------------------------------------------------------------+
void OpenBuyTrade(double atr, double confidence)
{
    MqlTick tick;
    if(!SymbolInfoTick(_Symbol, tick)) return;
    
    double tp = tick.ask + (atr * tp_multiplier);
    double sl = tick.ask - (atr * sl_multiplier);
    
    string comment = StringFormat("ML_BUY_%.2f", confidence);
    
    if(m_trade.Buy(lotsize, _Symbol, tick.ask, sl, tp, comment))
    {
        Print("✅ BUY opened: Confidence=", confidence, " TP=", tp, " SL=", sl);
    }
    else
    {
        Print("❌ BUY failed: ", GetLastError());
    }
}

//+------------------------------------------------------------------+
//| Open Sell Trade                                                  |
//+------------------------------------------------------------------+
void OpenSellTrade(double atr, double confidence)
{
    MqlTick tick;
    if(!SymbolInfoTick(_Symbol, tick)) return;
    
    double tp = tick.bid - (atr * tp_multiplier);
    double sl = tick.bid + (atr * sl_multiplier);
    
    string comment = StringFormat("ML_SELL_%.2f", confidence);
    
    if(m_trade.Sell(lotsize, _Symbol, tick.bid, sl, tp, comment))
    {
        Print("✅ SELL opened: Confidence=", confidence, " TP=", tp, " SL=", sl);
    }
    else
    {
        Print("❌ SELL failed: ", GetLastError());
    }
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    Print("🚀 Initializing XAUUSD ML Trading EA...");
    
    // First, let's check if the file exists and get more detailed error info
    Print("📁 Checking ONNX file: ", ONNXFilename);
    Print("📂 Expected location: MQL5/Files/", ONNXFilename);
    
    // Try to get file handle first to verify file exists
    int file_handle = FileOpen(ONNXFilename, FILE_READ|FILE_BIN);
    if(file_handle == INVALID_HANDLE)
    {
        Print("❌ Cannot access file: ", ONNXFilename);
        Print("📋 Last error: ", GetLastError());
        Print("💡 Possible solutions:");
        Print("   1. Check file permissions");
        Print("   2. Ensure file is not corrupted");
        Print("   3. Try copying file again to MQL5/Files/");
        Print("🔄 Continuing with fallback prediction mode");
        model_loaded = false;
        
        Print("⚠️ ONNX file not accessible - using simple technical analysis");
        Print("🎯 Fallback Mode: ", ModelName);
        Print("📊 Using basic momentum and trend analysis");
        Print("🔧 Confidence threshold: ", confidence_threshold);
        Print("💰 TP multiplier: ", tp_multiplier, "x ATR");
        Print("🛑 SL multiplier: ", sl_multiplier, "x ATR");
        
        return INIT_SUCCEEDED; // Continue with fallback mode
    }
    else
    {
        int file_size = (int)FileSize(file_handle);
        Print("✅ File found! Size: ", file_size, " bytes");
        FileClose(file_handle);
    }
    
    // Try loading ONNX model with detailed error reporting
    Print("🔄 Attempting to load ONNX model...");
    model_handle = OnnxCreate(ONNXFilename, ONNX_NO_CONVERSION);
    
    if(model_handle == INVALID_HANDLE)
    {
        int error_code = GetLastError();
        Print("❌ Failed to load ONNX model: ", ONNXFilename);
        Print("📋 Error code: ", error_code);
        
        // Common MT5 ONNX errors and solutions
        Print("💡 Common solutions:");
        Print("   1. Model compatibility: Ensure model was exported with opset <= 14");
        Print("   2. Model format: Try re-exporting model with explicit input/output names");
        Print("   3. Model operators: Check if model uses unsupported operators");
        Print("   4. File corruption: Re-export and copy the model file");
        Print("   5. Memory: Close other EAs and restart MT5");
        
        // Try alternative loading methods
        Print("🔄 Trying alternative loading with ONNX_DEFAULT...");
        model_handle = OnnxCreate(ONNXFilename, ONNX_DEFAULT);
        
        if(model_handle == INVALID_HANDLE)
        {
            Print("❌ Alternative loading also failed");
            Print("📋 Alternative error: ", GetLastError());
            Print("🔄 Continuing with fallback prediction mode");
            model_loaded = false;
            
            Print("⚠️ ONNX model could not be loaded - using simple technical analysis");
            Print("🎯 Fallback Mode: ", ModelName);
            Print("📊 Using basic momentum and trend analysis");
            Print("🔧 Confidence threshold: ", confidence_threshold);
            Print("💰 TP multiplier: ", tp_multiplier, "x ATR");
            Print("🛑 SL multiplier: ", sl_multiplier, "x ATR");
            
            return INIT_SUCCEEDED; // Continue with fallback mode
        }
        else
        {
            Print("✅ Alternative loading succeeded with ONNX_DEFAULT");
        }
    }
    else
    {
        Print("✅ ONNX model loaded successfully with ONNX_NO_CONVERSION");
    }
    
    // Verify model structure
    Print("🔍 Verifying model structure...");
    
    // Test with dummy input to ensure model works
    matrix test_input(1, 19);
    for(int i = 0; i < 19; i++)
        test_input[0][i] = 1.0; // Dummy values
    
    matrix test_output;
    bool test_success = OnnxRun(model_handle, ONNX_NO_CONVERSION, test_input, test_output);
    
    if(!test_success)
    {
        Print("❌ Model test run failed");
        Print("📋 Test error: ", GetLastError());
        Print("💡 Model might have incompatible input/output structure");
        Print("🔄 Continuing with fallback prediction mode");
        model_loaded = false;
        
        Print("⚠️ ONNX model test failed - using simple technical analysis");
        Print("🎯 Fallback Mode: ", ModelName);
        Print("📊 Using basic momentum and trend analysis");
        Print("🔧 Confidence threshold: ", confidence_threshold);
        Print("💰 TP multiplier: ", tp_multiplier, "x ATR");
        Print("🛑 SL multiplier: ", sl_multiplier, "x ATR");
        
        return INIT_SUCCEEDED; // Continue with fallback mode
    }
    else
    {
        Print("✅ Model test run successful");
        Print("📊 Output shape: ", test_output.Rows(), "x", test_output.Cols());
        if(test_output.Rows() > 0 && test_output.Cols() > 0)
        {
            Print("📈 Test output probability: ", test_output[0][0]);
            Print("📈 Test prediction: ", (test_output[0][0] > 0.5) ? "BUY" : "SELL");
        }
    }
    
    Print("✅ ONNX model loaded and verified successfully");
    model_loaded = true;
    Print("🎯 Model: ", ModelName);
    Print("📊 Features: 19");
    Print("🔧 Confidence threshold: ", confidence_threshold);
    Print("💰 TP multiplier: ", tp_multiplier, "x ATR");
    Print("🛑 SL multiplier: ", sl_multiplier, "x ATR");
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // Only trade at the start of new hour (model predicts next hour)
    if(!NewHour()) return;
    
    // Limit concurrent positions
    if(HasOpenPositions() && PositionsTotal() >= max_positions) return;
    
    // Get ML prediction
    long prediction;
    double confidence;
    
    if(!GetMLPrediction(prediction, confidence))
    {
        Print("⚠️ ML prediction failed");
        return;
    }
    
    // Get ATR for dynamic TP/SL
    int atr_handle = iATR(_Symbol, PERIOD_H1, atr_period);
    if(atr_handle == INVALID_HANDLE)
    {
        Print("⚠️ Invalid ATR handle");
        return;
    }
    
    // Get the actual ATR value from the buffer
    double atr_buffer[];
    if(CopyBuffer(atr_handle, 0, 1, 1, atr_buffer) <= 0)
    {
        Print("⚠️ Failed to get ATR buffer");
        return;
    }
    
    double atr = atr_buffer[0];
    if(atr <= 0)
    {
        Print("⚠️ Invalid ATR buffer value");
        return;
    }
    
    // Trading logic with confidence filtering
    if(prediction == 1 && confidence > confidence_threshold)
    {
        // High confidence BUY signal
        Print("🔵 ML Signal: BUY | Confidence: ", confidence);
        if(!HasOpenPositions())
            OpenBuyTrade(atr, confidence);
    }
    else if(prediction == 0 && confidence > confidence_threshold)
    {
        // High confidence SELL signal  
        Print("🔴 ML Signal: SELL | Confidence: ", confidence);
        if(!HasOpenPositions())
            OpenSellTrade(atr, confidence);
    }
    else
    {
        // Low confidence - skip trade
        string signal_text = (prediction == 1) ? "BUY" : "SELL";
        Print("⚪ ML Signal: ", signal_text, " | Confidence: ", confidence, " (SKIPPED - Low confidence)");
    }
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    if(model_handle != INVALID_HANDLE)
    {
        OnnxRelease(model_handle);
        Print("🔧 ONNX model released");
    }
}