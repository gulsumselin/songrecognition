Bu çalışmanın amacı, ses dosyalarından çıkarılan öznitelikler ile sanatçı tahmini yapabilecek bir sınıflandırma modeli geliştirmektir. Bu kapsamda, aşağıdaki hedefler gerçekleştirilmiştir:
1.	Deezer API kullanılarak farklı sanatçılara ait ses dosyalarının toplanması.
2.	Bu dosyalardan akustik özniteliklerin çıkarılması ve analiz edilmesi.
3.	Çıkarılan özniteliklerin normalize edilerek model eğitimi için uygun hale getirilmesi.
4.	CNN tabanlı bir model geliştirilerek sanatçı tahmini doğruluğunun artırılması.
5.	Elde edilen sonuçların detaylı bir şekilde analiz edilerek, modelin etkinliğinin değerlendirilmesi.
________________________________________

Bu çalışmada, dijital müzik platformlarında şarkı tanıma süreçlerini hızlandırmak ve doğruluğu artırmak amacıyla bir sınıflandırma modeli oluşturulmuştur. Çalışma kapsamında:
Veri toplama: Deezer API kullanılarak farklı sanatçılara ait toplamda 3022 şarkı bilgisi toplanmış ve bu şarkıların ön dinleme URL'leri WAV formatında kaydedilmiştir.
Özellik çıkarma: Kaydedilen ses dosyalarından MFCC, Chroma, Spectral Contrast, Spectral Bandwidth ve Zero Crossing Rate (ZCR) gibi öznitelikler çıkarılmıştır.
Veri normalizasyonu: Çıkarılan özellikler MinMaxScaler ile normalize edilerek model eğitimi için uygun hale getirilmiştir.
Model geliştirme: Normalize edilmiş veriler bir CNN modeli ile sınıflandırılmıştır. Model, 141 farklı sanatçıya ait şarkıları tahmin edebilme kapasitesine sahiptir.
