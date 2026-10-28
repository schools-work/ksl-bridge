-- Demo venue: a restaurant with a menu knowledge base, venue vocabulary and enabled tools (DEV / demos).
INSERT INTO venue.venues (id, name, venue_type, owner_id, county_code, address, settings)
VALUES ('00000000-0000-4000-8000-00000000a001', 'Mama Oliech Demo Kitchen', 'restaurant',
        '00000000-0000-4000-8000-0000000000d1', '047', 'Nairobi', '{"upsell_drink": true, "clarify_cost": 0.15}')
ON CONFLICT (id) DO NOTHING;

INSERT INTO venue.staff (venue_id, user_id, role) VALUES
  ('00000000-0000-4000-8000-00000000a001', '00000000-0000-4000-8000-0000000000d1', 'manager')
ON CONFLICT DO NOTHING;

INSERT INTO lexicon.glosses (gloss, categories, status) VALUES
  ('BURGER', '{FOOD}', 'pending'), ('CHICKEN', '{FOOD}', 'pending'), ('CHIPS', '{FOOD}', 'pending'),
  ('CHAPATI', '{FOOD}', 'pending'), ('UGALI', '{FOOD}', 'pending'), ('SODA', '{DRINK}', 'pending'),
  ('TEA', '{DRINK}', 'pending'), ('JUICE', '{DRINK}', 'pending'), ('PAY', '{PAYMENT}', 'pending'),
  ('TOILET', '{PLACE}', 'pending'), ('BILL', '{PAYMENT}', 'pending')
ON CONFLICT (gloss) DO NOTHING;

INSERT INTO lexicon.translations (gloss, language, phrase, is_primary) VALUES
  ('WATER', 'en', 'water', true), ('FOOD', 'en', 'food', true), ('BURGER', 'en', 'burger', true),
  ('CHICKEN', 'en', 'chicken', true), ('CHIPS', 'en', 'chips', true), ('CHIPS', 'en', 'fries', false),
  ('SODA', 'en', 'soda', true), ('SODA', 'en', 'soft drink', false), ('TOILET', 'en', 'toilet', true),
  ('TOILET', 'en', 'bathroom', false), ('TOILET', 'en', 'washroom', false), ('TOILET', 'sw', 'choo', true),
  ('WATER', 'sw', 'maji', true), ('TEA', 'en', 'tea', true), ('TEA', 'sw', 'chai', true)
ON CONFLICT DO NOTHING;

INSERT INTO knowledge.documents (id, venue_id, kind, title, content_sha256)
VALUES ('00000000-0000-4000-8000-00000000d001', '00000000-0000-4000-8000-00000000a001', 'menu', 'Menu', digest('menu-v1', 'sha256')),
       ('00000000-0000-4000-8000-00000000d002', '00000000-0000-4000-8000-00000000a001', 'directions', 'Directions', digest('dir-v1', 'sha256'))
ON CONFLICT (id) DO NOTHING;

INSERT INTO knowledge.chunks (document_id, section_path, ordinal, content, meta) VALUES
  ('00000000-0000-4000-8000-00000000d001', 'Menu > Food',   0, 'Menu item: chicken burger. Category: food. Price: KSh 650.', '{"item_id":17,"price":650}'),
  ('00000000-0000-4000-8000-00000000d001', 'Menu > Food',   1, 'Menu item: chips. Category: food. Price: KSh 250.',          '{"item_id":3,"price":250}'),
  ('00000000-0000-4000-8000-00000000d001', 'Menu > Food',   2, 'Menu item: ugali beef. Category: food. Price: KSh 450.',     '{"item_id":30,"price":450}'),
  ('00000000-0000-4000-8000-00000000d001', 'Menu > Food',   3, 'Menu item: chapati. Category: food. Price: KSh 50.',         '{"item_id":31,"price":50}'),
  ('00000000-0000-4000-8000-00000000d001', 'Menu > Drinks', 4, 'Menu item: water. Category: drink. Price: KSh 100.',         '{"item_id":21,"price":100}'),
  ('00000000-0000-4000-8000-00000000d001', 'Menu > Drinks', 5, 'Menu item: soda. Category: drink. Price: KSh 120.',          '{"item_id":22,"price":120}'),
  ('00000000-0000-4000-8000-00000000d002', 'Directions',    0, 'The toilet is at the back of the restaurant, to the left of the kitchen.', '{}'),
  ('00000000-0000-4000-8000-00000000d002', 'Directions',    1, 'You can pay with cash or M-Pesa at the counter near the entrance.', '{}')
ON CONFLICT DO NOTHING;

INSERT INTO knowledge.venue_vocabulary (venue_id, term, gloss, category, bias_nats) VALUES
  ('00000000-0000-4000-8000-00000000a001', 'chicken burger', 'BURGER',  'FOOD',  0.8),
  ('00000000-0000-4000-8000-00000000a001', 'chips',          'CHIPS',   'FOOD',  0.8),
  ('00000000-0000-4000-8000-00000000a001', 'ugali beef',     'UGALI',   'FOOD',  0.8),
  ('00000000-0000-4000-8000-00000000a001', 'chapati',        'CHAPATI', 'FOOD',  0.8),
  ('00000000-0000-4000-8000-00000000a001', 'water',          'WATER',   'DRINK', 0.8),
  ('00000000-0000-4000-8000-00000000a001', 'soda',           'SODA',    'DRINK', 0.8),
  ('00000000-0000-4000-8000-00000000a001', 'Oliech',         NULL,      'PERSON', 0.5)
ON CONFLICT DO NOTHING;

INSERT INTO agent.venue_tools (venue_id, tool_id, adapter)
SELECT '00000000-0000-4000-8000-00000000a001', id, 'demo_restaurant' FROM agent.tools
WHERE name IN ('menu.lookup', 'order.add', 'kb.answer', 'staff.call', 'ui.show_menu')
ON CONFLICT DO NOTHING;

INSERT INTO agent.definitions (name, version, venue_type, intents, slot_prompts, policy) VALUES
  ('restaurant-ordering', 1, 'restaurant',
   '{"order_food": ["item", "qty"], "ask_location": ["place"], "pay": ["method"], "greet": [], "help": []}',
   '{"item": {"text": "What would you like?", "expect": ["FOOD", "DRINK"]},
     "qty": {"text": "How many?", "expect": ["NUMBER"]},
     "place": {"text": "Where do you want to go?", "expect": ["PLACE"]},
     "method": {"text": "Cash or M-Pesa?", "expect": ["PAYMENT"]}}',
   '{"min_intent_confidence": 0.4, "upsell_drink": true, "escalate_after_failed_turns": 2}')
ON CONFLICT DO NOTHING;
