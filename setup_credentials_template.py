"""
MT5 Credentials Setup Template
Edit this file with your MT5 details and run it ONCE to save encrypted credentials
"""

from xauusd_real_tick_collection import XAUUSDTickCollector

def setup_my_credentials():
    """
    REPLACE THE VALUES BELOW WITH YOUR ACTUAL MT5 CREDENTIALS
    """
    
    # YOUR MT5 CREDENTIALS (REPLACE THESE!)
    my_login = "205169129"              # Your MT5 account number  
    my_password = "Amm@gg180262"    # Your MT5 password
    my_server = "Exness-MT5Trial7"      # Your broker's server name
    account_name = "205169129"    # Name for your saved credentials
    
    print("SETTING UP YOUR MT5 CREDENTIALS")
    print("=" * 50)
    print("IMPORTANT: Only run this script ONCE to save credentials")
    print("After running once, use the main collection script")
    
    collector = XAUUSDTickCollector()
    
    result = collector.setup_credentials(
        login=my_login,
        password=my_password,
        server=my_server,
        account_name=account_name
    )
    
    if result:
        print(f"\nSUCCESS! Credentials saved as: {account_name}")
        print("Your credentials are now encrypted and secure")
        print("\nNEXT STEPS:")
        print("   1. Edit xauusd_real_tick_collection.py")
        print(f"   2. Set account_name = '{account_name}'")
        print("   3. Run the main collection script")
        print("\nYou can delete this script after running it once")
    else:
        print("\nFAILED to save credentials")
        print("Check your MT5 details and try again")

if __name__ == "__main__":
    print("BEFORE RUNNING:")
    print("   1. Replace the credentials above with YOUR actual MT5 details")
    print("   2. Find your server name in MT5 -> Tools -> Options -> Server")
    print("   3. Make sure MT5 terminal is installed and working")
    print("\n   Press Enter to continue or Ctrl+C to exit...")
    input()
    
    setup_my_credentials() 