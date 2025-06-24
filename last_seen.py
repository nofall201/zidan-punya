# last_seen.py (Versi yang sudah diperbaiki)

# 1. Import 'app' dari file utama Anda (asumsi nama filenya app.py)
#    Ini penting untuk mendapatkan akses ke 'mesin' Flask.
from app import app, db, User, datetime

# 2. Buat dan aktifkan application context menggunakan 'with'
#    Semua kode yang butuh akses ke database harus berada di dalam blok ini.
with app.app_context():
    print("Berhasil masuk ke dalam application context. Memulai proses...")
    
    # 3. Letakkan semua logika database Anda di dalam blok 'with' ini
    
    # Cari semua user yang kolom last_seen-nya masih kosong (NULL)
    print("Mencari pengguna dengan last_seen kosong...")
    users_to_update = User.query.filter(User.last_seen == None).all()
    print(f"Ditemukan {len(users_to_update)} pengguna untuk diupdate.")

    if users_to_update:
        # Loop setiap user dan atur last_seen = created_at
        for user in users_to_update:
            if user.created_at:
                user.last_seen = user.created_at
            else:
                # Pengaman jika created_at juga kosong, isi dengan waktu sekarang
                user.last_seen = datetime.utcnow()
            print(f"Mengupdate pengguna: {user.username}...")

        # Simpan semua perubahan ke database
        db.session.commit()
        print("\nSUCCESS: Semua perubahan berhasil disimpan ke database!")
    else:
        print("\nINFO: Tidak ada pengguna yang perlu diupdate.")

print("Proses selesai, context ditutup.")