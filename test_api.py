import json
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

print("=" * 65)
print("TEST 1: Health Check (GET /)")
print("=" * 65)
res_health = client.get("/")
print(f"HTTP Status: {res_health.status_code}")
print(f"Response: {json.dumps(res_health.json(), indent=2)}")

print("\n" + "=" * 65)
print("TEST 2: POST /api/webhook/message (Mock WhatsApp Interaction)")
print("=" * 65)

# Mocking a real-world Markdown file content passed as user status
mock_md_status = """# Street Vendor Onboarding Checklist
- **Vendor ID**: SV-KL-2026-881
- **Current Stage**: Document Verification
- **Verified Items**:
  - [x] Aadhaar Card Verified
  - [x] Passport Photo Uploaded
  - [ ] Food Safety Training Certificate (Pending)
- **Vendor Category**: Mobile Food Cart (Snacks & Tea)
- **Assigned Officer**: Ward 12 Inspector
"""

mock_request_payload = {
    "user_id": "whatsapp:+919876543210",
    "status": mock_md_status,
    "need": "What documents are required for FSSAI registration of a tea stall?",
    "session_id": "sess_9876543210_2026",
    "metadata": {
        "platform": "whatsapp",
        "sender_name": "Ramesh Kumar",
        "locale": "ml-IN"
    }
}

print("Incoming Payload from Workflow A:")
print(json.dumps(mock_request_payload, indent=2))
print("-" * 65)

res_webhook = client.post("/api/webhook/message", json=mock_request_payload)
print(f"HTTP Status: {res_webhook.status_code}")
print("Response returned to Workflow A:")
print(json.dumps(res_webhook.json(), indent=2))

print("\n" + "=" * 65)
print("TEST 3: Schema Validation (Missing required field 'need')")
print("=" * 65)
invalid_payload = {
    "user_id": "whatsapp:+919876543210",
    "status": "In progress"
    # 'need' is intentionally missing
}
res_invalid = client.post("/api/webhook/message", json=invalid_payload)
print(f"HTTP Status: {res_invalid.status_code} (Expected 422 Unprocessable Entity)")
print(f"Validation Error: {res_invalid.json().get('detail')[0].get('msg')}")
print("=" * 65)
