from typing import List, Tuple, Optional
import re

class TokenDecoder:
    def __init__(self):
        self.encoding_groups = {
            'turkish': [
                'utf-8',
                'iso-8859-9',  # Latin-5, специально для турецкого
                'cp857',       # DOS Turkish
                'cp1254'       # Windows Turkish
            ],
            'greek': [
                'utf-8',
                'iso-8859-7',
                'cp1253',
                'cp737'
            ],
            'cyrillic': [
                'utf-8',
                'windows-1251',
                'cp1251',
                'koi8-r',
                'cp866'
            ],
            'latin': [
                'utf-8',
                'latin1',
                'iso-8859-1',
                'cp1252'
            ]
        }

    def is_turkish_text(self, text: str) -> bool:
        """
        Проверяет наличие специфических турецких символов
        """
        turkish_chars = set('İıĞğÜüŞşÖöÇç')
        return any(c in turkish_chars for c in text)

    def is_greek_text(self, text: str) -> bool:
        """
        Проверяет наличие греческих букв
        """
        return bool(re.search('[\u0370-\u03FF\u1F00-\u1FFF]', text))

    def is_russian_text(self, text: str) -> bool:
        """
        Проверяет наличие русских букв
        """
        return bool(re.search('[а-яА-Я]', text))

    def _try_decode_turkish(self, encoded_text: str) -> Optional[Tuple[str, str]]:
        """
        Специализированное декодирование для турецкого текста
        """
        try:
            for encoding in ['utf-8', 'iso-8859-9', 'cp1254', 'cp857']:
                try:
                    bytes_data = encoded_text.encode('latin1', errors='ignore')
                    decoded = bytes_data.decode(encoding, errors='ignore')
                    if self.is_turkish_text(decoded):
                        return (encoding, decoded)
                except:
                    continue
        except Exception as e:
            print(f"Ошибка при декодировании турецкого текста: {str(e)}")
        return None

    def decode_token(self, encoded_text: str) -> List[Tuple[str, str, str]]:
        """
        Декодирует токен всеми доступными способами
        """
        results = []
        
        if encoded_text.startswith('Ġ'):
            encoded_text = encoded_text[1:]

        # Сначала пробуем специальное декодирование для турецкого текста
        turkish_result = self._try_decode_turkish(encoded_text)
        if turkish_result:
            encoding, decoded = turkish_result
            results.append(('turkish', encoding, decoded))

        # Затем пробуем все остальные кодировки
        for language_group, encodings in self.encoding_groups.items():
            for encoding in encodings:
                try:
                    bytes_data = encoded_text.encode('latin1', errors='ignore')
                    decoded = bytes_data.decode(encoding, errors='ignore')
                    
                    if (decoded and decoded != encoded_text and 
                        not all(c in '??□' for c in decoded)):
                        results.append((language_group, encoding, decoded))
                except:
                    continue

        # Удаляем дубликаты с сохранением порядка
        seen = set()
        unique_results = []
        for result in results:
            if result[2] not in seen:
                seen.add(result[2])
                unique_results.append(result)

        return unique_results

    def print_results(self, results: List[Tuple[str, str, str]]) -> None:
        """
        Форматированный вывод результатов декодирования
        """
        if not results:
            print("Не удалось декодировать строку")
            return

        print("\nРезультаты декодирования:")
        print("-" * 80)
        print(f"{'Группа':<15} {'Кодировка':<12} {'Результат':<50}")
        print("-" * 80)
        
        for language_group, encoding, decoded in results:
            # Маркеры для разных языков
            turkish_marker = "T" if self.is_turkish_text(decoded) else " "
            greek_marker = "G" if self.is_greek_text(decoded) else " "
            russian_marker = "R" if self.is_russian_text(decoded) else " "
            marker = f"{turkish_marker}{greek_marker}{russian_marker}"
            
            # Добавляем байтовое представление для отладки
            bytes_repr = ' '.join(f'{b:02x}' for b in decoded.encode('utf-8', errors='ignore'))
            print(f"{marker}{language_group:<12} {encoding:<12} {decoded:<30} [{bytes_repr}]")

if __name__ == "__main__":
    test_tokens = [
        'Ð¾Ð¿', 'ÑĢÐ¾Ñģ', ':', 'ĠÐļÐ°Ðº', 'Ð¸Ð¼Ð¸', 'ĠÑģÑĢÐµÐ´ÑģÑĤÐ²', 'Ð°Ð¼Ð'
    ]

    decoder = TokenDecoder()
    for token in test_tokens:
        print(f"\nИсходный токен: {token}")
        results = decoder.decode_token(token)
        decoder.print_results(results)