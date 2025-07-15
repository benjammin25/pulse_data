from datetime import datetime
import pandas as pd 

def create_price_changes_from_user_input():
    """
    Interactive function to create price_changes list from user input
    First asks if there are any price changes at all
    """
    
    print("💰 PRICE CHANGE CONFIGURATION")
    print("="*35)
    
    # Initial question about price changes
    while True:
        has_changes = input("Are there any price changes to apply? (y/n): ").strip().lower()
        if has_changes in ['y', 'yes', 'n', 'no']:
            break
        print("❌ Please enter 'y' for yes or 'n' for no")
    
    # If no price changes, return empty list
    if has_changes in ['n', 'no']:
        print("✅ No price changes will be applied.")
        return []
    
    # If yes, proceed with price change entry
    price_changes = []
    print("\n📝 Enter price changes one by one. Press Enter with empty package name to finish.")
    
    while True:
        print(f"\n--- Price Change #{len(price_changes) + 1} ---")
        
        # Get package name
        package = input("Package name (or press Enter to finish): ").strip()
        if not package:
            break
        
        # Get effective date
        while True:
            date_str = input("Effective date (YYYY-MM-DD): ").strip()
            try:
                # Validate date format
                datetime.strptime(date_str, '%Y-%m-%d')
                break
            except ValueError:
                print("❌ Invalid date format. Please use YYYY-MM-DD")
        
        # Get old price
        while True:
            try:
                old_price = float(input("Old price ($): ").strip())
                break
            except ValueError:
                print("❌ Invalid price. Please enter a number")
        
        # Get new price
        while True:
            try:
                new_price = float(input("New price ($): ").strip())
                break
            except ValueError:
                print("❌ Invalid price. Please enter a number")
        
        # Add to list
        price_change = {
            'package': package,
            'effective_date': date_str,
            'old_price': old_price,
            'new_price': new_price
        }
        
        price_changes.append(price_change)
        
        print(f"✅ Added: {package} from ${old_price} to ${new_price} on {date_str}")
        
        # Ask if they want to add another
        while True:
            add_another = input("\nAdd another price change? (y/n): ").strip().lower()
            if add_another in ['y', 'yes', 'n', 'no']:
                break
            print("❌ Please enter 'y' for yes or 'n' for no")
        
        if add_another in ['n', 'no']:
            break
    
    # Summary
    if price_changes:
        print(f"\n🎯 TOTAL PRICE CHANGES CONFIGURED: {len(price_changes)}")
        for i, change in enumerate(price_changes, 1):
            print(f"   {i}. {change['package']}: ${change['old_price']} → ${change['new_price']} ({change['effective_date']})")
    else:
        print("\n✅ No price changes configured.")
    
    return price_changes


def apply_price_changes_to_new_contracts(df, price_changes):
    """
    Apply price changes to contracts that START on or after the effective date
    (This handles the cases your splitting function misses)
    """
    
    df = df.copy()
    df['Package Start Date'] = pd.to_datetime(df['Package Start Date'])
    
    total_changes = 0
    
    print("📅 APPLYING PRICE CHANGES TO NEW CONTRACTS")
    print("="*50)
    
    for change in price_changes:
        package = change['package']
        effective_date = pd.to_datetime(change['effective_date'])
        old_price = change['old_price']
        new_price = change['new_price']
        # Debug version
        print(f"Looking for: Package='{package}', Price={old_price}, Date>={effective_date}")

        # Check each condition separately
        pkg_match = df['Package Group Condensed'] == package
        date_match = df['Package Start Date'] >= effective_date  
        price_match = df['Price'] == old_price

        print(f"Package matches: {pkg_match.sum()}")
        print(f"Date matches: {date_match.sum()}")  
        print(f"Price matches: {price_match.sum()}")
        print(f"All three match: {(pkg_match & date_match & price_match).sum()}")
        # Find contracts that START on or after the price change date
        mask = (
            (df['Package Group Condensed'] == package) &
            (df['Package Start Date'] >= effective_date) &  # Key difference: >= not 
            (df['Price'] == old_price)
        )
        
        # Apply the new price
        changes_made = mask.sum()
        df.loc[mask, 'Price'] = new_price
        total_changes += changes_made
        
        print(f"📦 {package}: Updated {changes_made} contracts starting from {effective_date.strftime('%Y-%m-%d')}")
        print(f"   Price: ${old_price} → ${new_price}")
    
    print(f"\n✅ Total new contracts updated: {total_changes}")
    return df


def split_contracts_for_price_changes(df, price_changes):
    """
    Split customer contracts when price changes occur mid-contract
    
    Example: Customer with June 1 - August 1 contract, price change July 1
    Results in: 
    - Row 1: June 1 - June 30 (old price)
    - Row 2: July 1 - August 1 (new price)
    """
    
    df = df.copy()
    df['Package Start Date'] = pd.to_datetime(df['Package Start Date'])
    # Fix the 2999 date issue
    def safe_datetime_conversion(date_series):
        """Safely convert dates, handling year 2999"""
        result = []
        for date in date_series:
            if pd.isna(date):
                result.append(pd.NaT)
            elif str(date).startswith('2999') or '2999' in str(date):
                # Convert 2999 dates to a far future date that pandas can handle
                result.append(pd.Timestamp('2099-12-31'))
            else:
                try:
                    result.append(pd.to_datetime(date))
                except:
                    result.append(pd.NaT)
        return pd.Series(result)
    
    # Apply safe conversion to Package End Date
    df['Package End Date'] = safe_datetime_conversion(df['Package End Date'])
    
    new_rows = []
    rows_to_remove = []
    
    print("✂️  SPLITTING CONTRACTS FOR MID-CONTRACT PRICE CHANGES")
    print("="*60)
    
    for idx, row in df.iterrows():
        contract_start = row['Package Start Date']
        contract_end = row['Package End Date']
        package = row['Package Group Condensed']
        current_price = row['Price']
        
        # Find applicable price changes for this package
        applicable_changes = [
            change for change in price_changes 
            if change['package'] == package and change['old_price'] == current_price
        ]
        
        if not applicable_changes:
            # No price changes affect this contract
            continue
            
        # Sort price changes by effective date
        applicable_changes.sort(key=lambda x: pd.to_datetime(x['effective_date']))
        
        # Check if any price change falls within the contract period
        splits_needed = []
        for change in applicable_changes:
            effective_date = pd.to_datetime(change['effective_date'])
            if contract_start < effective_date < contract_end:
                splits_needed.append(change)
        
        if not splits_needed:
            # Price changes don't fall within this contract period
            continue
        """ 
        print(f"\n📋 Processing Contract:")
        print(f"   Account: {row['Account Number']}")
        print(f"   Package: {package}")
        print(f"   Original Period: {contract_start.strftime('%Y-%m-%d')} to {contract_end.strftime('%Y-%m-%d')}")
        print(f"   Original Price: ${current_price}")
        """
        # Mark original row for removal
        rows_to_remove.append(idx)
        
        # Create split periods
        current_start = contract_start
        current_price_value = current_price

        new_rows_for_this_contract = []

        for i, change in enumerate(splits_needed):
            effective_date = pd.to_datetime(change['effective_date'])
            new_price = change['new_price']
            
            # Create period before price change (if any)
            if current_start < effective_date:
                period_end = effective_date - pd.Timedelta(days=1)
            
                 #FIX 3: Ensure valid period (start <= end)
                if period_end >= current_start:
                    new_row = row.copy()
                    new_row['Package Start Date'] = current_start
                    new_row['Package End Date'] = period_end
                    new_row['Price'] = current_price_value
                    new_rows.append(new_row)
                    
                    # FIX 4: Add validation logging
                    period_days = (period_end - current_start).days + 1
                    if period_days <= 0:
                        print(f"⚠️  WARNING: Invalid period for account {row['Account Number']}: {period_days} days")
            
            # Update for next period
            current_start = effective_date
            current_price_value = new_price
        
        # Create final period with new price
        if current_start <= contract_end:
            new_row = row.copy()
            new_row['Package Start Date'] = current_start
            new_row['Package End Date'] = contract_end
            new_row['Price'] = current_price_value
            new_rows.append(new_row)
            #print(f"   ➡️  Period 2: {current_start.strftime('%Y-%m-%d')} to {contract_end.strftime('%Y-%m-%d')} @ ${current_price_value}")


    # Remove original rows and add split rows
    df_updated = df.drop(rows_to_remove).copy()
    
    if new_rows:
        new_rows_df = pd.DataFrame(new_rows)
        df_updated = pd.concat([df_updated, new_rows_df], ignore_index=True)
        df_updated = df_updated.sort_values(['Account Number', 'Package Start Date']).reset_index(drop=True)
    
    print(f"\n✅ SPLIT SUMMARY:")
    print(f"   Original contracts affected: {len(rows_to_remove)}")
    print(f"   New contract periods created: {len(new_rows)}")
    print(f"   Total contracts now: {len(df_updated)}")
    
    return df_updated