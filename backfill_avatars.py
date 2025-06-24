# Script untuk mengisi avatar pengguna lama (jalankan sekali)
from app import app, db, User, generate_avatar_url

with app.app_context():
    # Ambil semua user yang belum punya avatar_url
    users_to_update = User.query.filter(User.avatar_url == None).all()
    
    if not users_to_update:
        print("Semua pengguna sudah memiliki avatar. Tidak ada yang perlu diupdate.")
    else:
        print(f"Menemukan {len(users_to_update)} pengguna untuk diupdate...")
        
        for user in users_to_update:
            # Pastikan user punya gender sebelum membuat avatar
            if user.gender:
                new_avatar_url = generate_avatar_url(user)
                user.avatar_url = new_avatar_url
                print(f"Membuat avatar untuk user ID {user.id} ({user.username}) -> {new_avatar_url}")
            else:
                print(f"Skipping user ID {user.id} karena gender belum diatur.")
                
        # Commit semua perubahan ke database
        db.session.commit()
        print("Update selesai!")