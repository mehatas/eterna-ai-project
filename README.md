# Derin Öğrenme ve Makine Öğrenmesi Modelleri ile Toprak Verim Analizi

## 🚩 İçerikler
- [Genel Bakış](#Genel-Bakış)
- [Kullanılan Teknolojiler](#Kullanılan-Teknolojiler)
- [Projede Yapılan İşlemler](#Projede-Yapılan-İşlemler)
  * [Veri Önişleme ve Analizi](#Veri-Önişleme-ve-Analizi)
  * [K-Nearest Neighbors Classifier](#K-Nearest-Neighbors-Classifier)
  * [Tensorflow Deep Learning Classifier](#Tensorflow-Deep-Learning-Classifier)
  * [eXtreme Gradient Boosting Classifier](#eXtreme-Gradient-Boosting-Classifier)
- [Sonuç](#Sonuç)
- [Kaynaklar](#Kaynaklar)
- [İletişim](#İletişim)


## Genel Bakış
 Bu projede makine öğrenmesi algoritmaları ve derin öğrenme teknikleri kullanılarak toprak verim analizi gerçekleştirecek model eğitimleri gerçekleştirilmiştir. Uygulamada makine öğrenmesi tarafında K-Nearest Neighbors  ve eXtreme Gradient Boosting sınıflandırıcı kullanılırken, derin öğrenme modeli için tensorflow ve keras kütüphanelerinden yararlanılmıştır. 

## Kullanılan Teknolojiler
- **Platform**: Google Colab
- **Programlama Dili**: Python
- **Makine Öğrenmesi ve Deep Learning Kütüphaneleri**: Keras, Tensorflow, Scikit-Learn, XGboost
- **Makine Öğrenmesi Algoritmaları**: K-Nearest Neighbors Classifier, eXtreme Gradient Boosting Classifier

## Projede Yapılan İşlemler

### Veri Önişleme ve Analizi

Veri Önişleme ve Analizi aşamasında veri setinin okunması, veri setindeki özelliklerin incelenmesi, eksik verilerin silinmesi/tamamlanması, korelasyon analizi, veri setinin test ve eğitim verisine bölünmesi gibi işlemler yapılmıştır. Bu işlemlerin yapılma nedeni veri setini anlamak, modellerin eğitim sırasında tutatlı sonuçlar vermesi için gerekli önişleme adımlarının yapılmasıdır.

#### Korelasyon Analizi

Korelasyon katsayısı, iki değişken arasındaki ilişkinin yönünü ve gücünü/kuvvetini ifade eder. Katsayı değeri -1 ile +1 arasında bir değer almaktadır. Korelasyon katsayısı 1’e veya -1’e yaklaştıkça, doğrusal ilişkinin kuvveti artarken, uzaklaştıkça ilişkinin kuvveti azalır. Korelasyon matrisi incelendiğinde bağımsız değişkenler arasında kuvvetli bir ilişki olmadığı gözlemlenmiştir. 


<p align="center">
  <img src="https://github.com/user-attachments/assets/d1f8b5f8-e996-4b69-9449-bc507f6ae604" alt="Image 1" width="600">
  <br>
  <b>Korelasyon Analizi</b>
</p>

### K-Nearest Neighbors Classifier
K-Nearest Neighbors Classifier (KNN), sınıflandırılma problemlerinde kullanılan denetimli bir makine öğrenmesi algoritmasıdır. Çalışma mantığı, sınıfı tahmin edilmesi istenen bir veriyi, eğitim verisindeki en yakın k komşusuna bakarak sınıflandırmaktır. Bu komşular genellikle Oklid uzaklığı gibi bir metrikle belirlenmektedir ve en yakın k komşu içerisinde yer alan çoğunluk sınıf tahmin edilen sınıf olarak kabul edilmektedir. KNN modeli eğitim sırasında tüm veriyi hafızada tutarak tahmin yapar, bu da onu büyük veri setlerinde yavaşlatabilir. Verilerin doğru şekilde ölçeklendirilmesi önemlidir, çünkü algoritma mesafe hesaplarına dayanmaktadır.

### Tensorflow Deep Learning Classifier
TensorFlow Deep Learning Classifier, sayısal veriler üzerinde ikili sınıflandırma problemlerinde güçlü bir şekilde kullanılmaktadır. Bu modeller, giriş olarak verilen sayısal özellikleri katmanlar aracılığıyla işlemekte ve her katmanda verinin daha soyut temsillerini öğrenmektedir. Son katmanda kullanılan sigmoid aktivasyon fonksiyonu 0 ile 1 arasında bir olasılık değeri üretir; bu da girdinin hangi sınıfa ait olduğuna dair karar vermede kullanılmaktadır. Model, binary cross-entropy gibi uygun bir kayıp fonksiyonuyla eğitilir ve doğruluk, precision, recall, f1-score, confusion matris gibi metriklerle değerlendirilmektedir. Sayısal veriler üzerinde doğru ön işleme (örneğin normalizasyon) yapıldığında ve büyük veri setleriyle bu yöntem, geleneksel sınıflandırma algoritmalarına kıyasla daha yüksek başarı sağlayabilmektedir.

### eXtreme Gradient Boosting Classifier
eXtreme Gradient Boosting Classifier, özellikle sınıflandırma ve regresyon problemlerinde yüksek doğruluk sağlayan gelişmiş bir gradient boosting algoritmasıdır. Zayıf tahmin edici olan karar ağaçlarını ardışık şekilde eğiterek, kendinden önceki ağaçların hatalarını minimize etmeye çalışır. Yani her yeni ağaç, önceki ağaçların yapamadığı tahmin hatalarını öğrenmeye çalışır ve bu sayede model zamanla daha iyi hale gelir. XGBoost, düzenleme (L1,L2 regularization) teknikleri içerdiği için overfittinge karşı da dayanıklıdır. Ayrıca eksik veriyle başa çıkabilme, paralel hesaplama yeteneği ve hız açısından avantajlı bir boosting algoritmasıdır.


## Sonuç
Proje kapsamında K-Nearest Neighbors, Deep Learning Model ve eXtreme Gradient Boosting algoritması kullanılarak 3 farklı model eğitilmiştir. Modeller aynı test verileri üzerinde accuracy, classification report(recall, precision,f1-score) ve confusion matris metrikleri ile değerlendirilmiştir. Bu değerlendirme metriklerinin incelenmesi sonucunda en iyi sonuç veren model %99.33'lük bir başarı oranıyla XGBoost Classifier olmuştur. Ayrıca veri setindeki sınıf dengesizliği, veri sayısının azlığı gibi durumlar göz önünde bulundurulduğunda; daha dengeli bir sınıf dağılımı bulunan büyük bir veri setiyle her modelin başarım oranının değişebileceği unutulmamalıdır. Halihazırda bulunan veri setine karşı modellerin sağladığı başarım oranı ve confusion matrix metrikleri aşağıdaki görselde gözükmektedir.


<p align="center">
  <img src="https://github.com/user-attachments/assets/4170e9ef-e56e-419b-ab2d-03862776a5e6" alt="Image 2" width="1000">
  <br>
  <b>Confusion Matrix</b>
</p>

Ayrıca seçilen XGBoost modeli üzerinde değişken önem analizi yapılarak modelin karar verme süreçlerinde değişkenlerin karar verme sürecine ortalama olarak ne oranda bilgi kazancı yaptığı hesaplanmıştır.


<p align="center">
  <img src="https://github.com/user-attachments/assets/98d797da-ee40-4b63-8ecd-1dbe318937f3" alt="Image 3" width="600">
  <br>
  <b>Features Importance</b>
</p>

## Kaynaklar
* https://scikit-learn.org/stable/index.html
* https://www.educative.io/answers/classification-using-xgboost-in-python
* https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier
* https://www.veribilimiokulu.com/xgboost-nasil-calisir/
* https://medium.com/@alifiya13/binary-classification-with-tensorflow-d36bfd0e4988
* https://www.freecodecamp.org/news/binary-classification-made-simple-with-tensorflow/
* https://www.tensorflow.org/api_docs/python/tf/keras/Model
* https://how.dev/answers/what-is-gradient-boosting
* https://ravenfo.com/2021/08/23/korelasyon-nedir-python-korelasyon-analizi/
* https://medium.com/@gulcanogundur/do%C4%9Fruluk-accuracy-kesinlik-precision-duyarl%C4%B1l%C4%B1k-recall-ya-da-f1-score-300c925feb38


## İletişim
- Mehmet Ataş  
  Email: atasmehmet@protonmail.com
