#!/usr/bin/env python3
"""
Validation script to check if the OS knowledge base integration is working correctly.
"""

import json
import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} missing: {filepath}")
        return False

def validate_json_structure(filepath, required_keys):
    """Validate JSON file structure"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) > 0:
            # Check first item has required keys
            first_item = data[0]
            if all(key in first_item for key in required_keys):
                print(f"✅ {filepath} structure is valid ({len(data)} items)")
                return True
            else:
                print(f"❌ {filepath} missing required keys: {required_keys}")
                return False
        else:
            print(f"❌ {filepath} is not a valid list or is empty")
            return False
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return False

def main():
    """Main validation function"""
    print("🔍 Validating OS Knowledge Base Integration\n")

    all_valid = True

    # Check config files
    print("📋 Checking Configuration Files:")
    all_valid &= check_file_exists("config/taxonomy.json", "Taxonomy config")
    all_valid &= check_file_exists("config/topic_rules.json", "Topic rules config")

    # Check raw data files  
    print("\n📂 Checking Raw Data Files:")
    all_valid &= check_file_exists("data/raw/complete_dbms.json", "DBMS questions")
    all_valid &= check_file_exists("data/raw/oops_qna_simplified.json", "OOPs questions") 
    all_valid &= check_file_exists("data/raw/os_qna.json", "OS questions")

    # Check scripts
    print("\n🔧 Checking Script Files:")
    all_valid &= check_file_exists("scripts/prepare_kb.py", "KB preparation script")
    all_valid &= check_file_exists("scripts/build_faiss_gemini.py", "FAISS build script")
    all_valid &= check_file_exists("scripts/rag_query.py", "RAG query script")

    # Validate JSON structures
    print("\n🔍 Validating File Structures:")
    if Path("config/taxonomy.json").exists():
        try:
            with open("config/taxonomy.json", 'r') as f:
                taxonomy = json.load(f)
            topics = [topic["name"] for topic in taxonomy["topics"]]
            if "OS" in topics:
                print("✅ OS topic found in taxonomy")
            else:
                print("❌ OS topic not found in taxonomy")
                all_valid = False
        except Exception as e:
            print(f"❌ Error validating taxonomy: {e}")
            all_valid = False

    if Path("data/raw/os_qna.json").exists():
        all_valid &= validate_json_structure("data/raw/os_qna.json", ["id", "question", "answer"])

    # Check for OS keywords in topic rules
    if Path("config/topic_rules.json").exists():
        try:
            with open("config/topic_rules.json", 'r') as f:
                rules = json.load(f)
            os_rules = [rule for rule in rules if rule["topic"] == "OS"]
            print(f"✅ Found {len(os_rules)} OS topic rules")
        except Exception as e:
            print(f"❌ Error checking topic rules: {e}")
            all_valid = False

    # Final status
    print("\n" + "="*50)
    if all_valid:
        print("🎉 All validations passed! The OS knowledge base integration is complete.")
        print("\n📋 Next Steps:")
        print("1. Set your Gemini API key: export GEMINI_API_KEY='your-key'")
        print("2. Run: python scripts/prepare_kb.py")
        print("3. Run: python scripts/build_faiss_gemini.py") 
        print("4. Run: python scripts/rag_query.py")
    else:
        print("❌ Some validations failed. Please check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
