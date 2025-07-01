"""
MT5 Credentials Setup Template
Edit this file with your MT5 details and run it ONCE to save encrypted credentials
"""

from mt5_login import encrypt_credentials
import os

def setup_my_credentials():
    """
    EDIT THE VALUES BELOW - REPLACE WITH YOUR ACTUAL MT5 ACCOUNT DETAILS
    """
    
    # REPLACE THESE PLACEHOLDER VALUES WITH YOUR ACTUAL MT5 CREDENTIALS:
    my_login = "YOUR_LOGIN_NUMBER"        # Replace with your MT5 account number (e.g., "12345678")
    my_password = "YOUR_PASSWORD"         # Replace with your MT5 account password
    my_server = "YOUR_BROKER_SERVER"      # Replace with your broker's server (e.g., "MetaQuotes-Demo")
    account_name = "MY_ACCOUNT"           # Replace with a name for your saved credentials (e.g., "demo_account")
    
    # CHECK: Make sure you've replaced the placeholder values above!
    if (my_login == "YOUR_LOGIN_NUMBER" or 
        my_password == "YOUR_PASSWORD" or 
        my_server == "YOUR_BROKER_SERVER"):
        print("❌ ERROR: You need to edit the credentials above first!")
        print("📝 Please replace the placeholder values with your actual MT5 details:")
        print("   • my_login: Your MT5 account number")
        print("   • my_password: Your MT5 account password") 
        print("   • my_server: Your broker's server name")
        print("   • account_name: A name for your saved credentials")
        return False
    
    print("SETTING UP YOUR MT5 CREDENTIALS")
    print("=" * 50)
    print("IMPORTANT: Only run this script ONCE to save credentials")
    print("After running once, use mt5_login.py to connect to MT5")
    
    # Create credentials directory if it doesn't exist
    credentials_dir = "credentials"
    if not os.path.exists(credentials_dir):
        os.makedirs(credentials_dir)
        print(f"📁 Created {credentials_dir} directory")
    
    # Save encrypted credentials
    credentials_path = f"{credentials_dir}\\{account_name}.txt"
    
    try:
        encrypt_credentials(
            login=my_login,
            password=my_password,
            server=my_server,
            filepath=credentials_path
        )
        
        print(f"\n✅ SUCCESS! Credentials saved as: {credentials_path}")
        print("Your credentials are now encrypted and secure")
        print("\nNEXT STEPS:")
        print("   1. Use mt5_login.py to connect to MT5")
        print(f"   2. Call login_mt5('{account_name}') in your scripts")
        print("   3. Your credentials are safely stored and encrypted")
        print("\nYou can delete this script after running it once")
        return True
        
    except Exception as e:
        print(f"\n❌ FAILED to save credentials: {e}")
        print("Check your MT5 details and try again")
        return False

if __name__ == "__main__":
    print("🔧 MT5 CREDENTIALS SETUP")
    print("=" * 30)
    print("BEFORE RUNNING, YOU MUST:")
    print("   1. Open this file in a text editor")
    print("   2. Find the credential variables (lines ~15-18)")
    print("   3. Replace YOUR_LOGIN_NUMBER with your MT5 account number")
    print("   4. Replace YOUR_PASSWORD with your MT5 password")
    print("   5. Replace YOUR_BROKER_SERVER with your server name")
    print("      (Find server in MT5: Tools → Options → Server)")
    print("   6. Replace MY_ACCOUNT with a name for these credentials")
    print("   7. Save the file and run this script")
    print("\n   Press Enter to continue or Ctrl+C to exit...")
    input()
    
    setup_my_credentials() 