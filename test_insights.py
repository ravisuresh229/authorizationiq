#!/usr/bin/env python3
"""
Test script for the insight generation logic
"""

def generate_insights(feature_importance, input_data):
    """Test the insight generation logic"""
    insights = []
    
    # Precompute unique words among all keys, only words of length >= 3
    key_words = {}
    word_counts = {}
    for key in input_data.keys():
        words = [w for w in key.lower().split('_') if len(w) >= 3]
        key_words[key] = words
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

    for feature_info in feature_importance:
        feature_name = feature_info["feature"]
        importance = feature_info["importance"]
        direction = feature_info["direction"]
        matched_key = None
        matched_value = None

        # 1. Exact value match (case-insensitive, exact)
        for key, value in input_data.items():
            if str(value).lower() == feature_name.lower().split()[-1]:
                matched_key = key
                matched_value = value
                break
            # Also allow value to appear as a whole word in feature name
            feature_words = feature_name.lower().split()
            if str(value).lower() in feature_words:
                matched_key = key
                matched_value = value
                break

        # 2. Full key match (case-insensitive, as whole word)
        if not matched_key:
            feature_words = feature_name.lower().split()
            for key in input_data.keys():
                if key.lower() in feature_words:
                    matched_key = key
                    matched_value = input_data[key]
                    break

        # 3. Unique word match (only words of length >= 3, as whole word)
        if not matched_key:
            feature_words = set(feature_name.lower().split())
            for key, words in key_words.items():
                for word in words:
                    if word_counts[word] == 1 and word in feature_words:
                        matched_key = key
                        matched_value = input_data[key]
                        break
                if matched_key:
                    break

        # 4. Fallback
        if matched_key and matched_value is not None:
            if direction == "positive":
                insight = f"✅ {feature_name} ({matched_key}: {matched_value}) is supporting approval"
            else:
                insight = f"⚠️ {feature_name} ({matched_key}: {matched_value}) may reduce approval chances"
        else:
            if direction == "positive":
                insight = f"✅ {feature_name} is supporting approval"
            else:
                insight = f"⚠️ {feature_name} may reduce approval chances"

        insights.append({
            "feature": feature_name,
            "importance": importance,
            "direction": direction,
            "insight": insight,
            "matched_key": matched_key,
            "matched_value": matched_value
        })
    return insights

def test_insights():
    """Test the insight generation with sample data"""
    
    # Sample feature importance data (similar to what the model returns)
    sample_feature_importance = [
        {"feature": "Procedure Code 61514", "importance": 0.85, "direction": "positive"},
        {"feature": "Diagnosis Code I63.9", "importance": -0.42, "direction": "negative"},
        {"feature": "Provider Specialty Neurology", "importance": 0.31, "direction": "positive"},
        {"feature": "Patient Age 45", "importance": 0.18, "direction": "positive"},
        {"feature": "Payer Medicare", "importance": -0.25, "direction": "negative"},
        {"feature": "Urgency Flag Y", "importance": 0.67, "direction": "positive"},
        {"feature": "Region South", "importance": 0.12, "direction": "positive"},
        {"feature": "Documentation Complete N", "importance": -0.89, "direction": "negative"}
    ]
    
    # Sample input data (similar to what users submit)
    sample_input_data = {
        "patient_age": 45,
        "patient_gender": "M",
        "procedure_code": "61514",
        "diagnosis_code": "I63.9",
        "provider_specialty": "Neurology",
        "payer": "Medicare",
        "urgency_flag": "Y",
        "documentation_complete": "N",
        "prior_denials_provider": 2,
        "region": "South"
    }
    
    print("🧪 Testing Insight Generation Logic")
    print("=" * 50)
    
    print("\n📊 Sample Feature Importance:")
    for feature in sample_feature_importance:
        print(f"  {feature['feature']}: {feature['importance']} ({feature['direction']})")
    
    print("\n📝 Sample Input Data:")
    for key, value in sample_input_data.items():
        print(f"  {key}: {value}")
    
    print("\n🔍 Generated Insights:")
    print("-" * 50)
    
    insights = generate_insights(sample_feature_importance, sample_input_data)
    
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight['insight']}")
        print(f"   Feature: {insight['feature']}")
        print(f"   Importance: {insight['importance']}")
        print(f"   Matched: {insight['matched_key']} = {insight['matched_value']}")
        print()
    
    # Test edge cases
    print("\n🧪 Testing Edge Cases:")
    print("-" * 30)
    
    # Test with no matches
    edge_feature = [{"feature": "Some Unrelated Feature", "importance": 0.5, "direction": "positive"}]
    edge_input = {"patient_age": 30}
    
    edge_insights = generate_insights(edge_feature, edge_input)
    print(f"Edge case - no match: {edge_insights[0]['insight']}")
    
    # Test with partial matches
    partial_feature = [{"feature": "Age Group", "importance": 0.3, "direction": "positive"}]
    partial_input = {"patient_age": 45}
    
    partial_insights = generate_insights(partial_feature, partial_input)
    print(f"Partial match: {partial_insights[0]['insight']}")
    
    print("\n✅ Insight generation test completed!")

if __name__ == "__main__":
    test_insights() 