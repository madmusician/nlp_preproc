extern crate arrayvec;
extern crate byteorder;
/// Common utilities for preprocessing in Natural Language Processing.
extern crate fnv;
extern crate lazy_static;
extern crate rayon;
extern crate rayon_hash;

mod util;

use fnv::{FnvHashMap as HashMap, FnvHashSet as HashSet};
use rayon::prelude::*;
use std::sync::atomic::AtomicUsize;
use std::sync::RwLock;
use std::{cmp::Eq, hash::Hash};
use util::AtomicInc;

/// Word ID, 32-bit should be enough
pub type WordId = u32;

pub use util::save_npy_with_iter;

/// Vocabulary statistics.
#[derive(Debug)]
pub struct VocabStat<'a, T: Eq + Hash + ?Sized + 'a> {
    setting: VocabStatSetting<'a, T>,
    counter: RwLock<rayon_hash::HashMap<&'a T, AtomicUsize, fnv::FnvBuildHasher>>,
    num_processed_sentences: AtomicUsize,
}

impl<'a, T> VocabStat<'a, T>
where
    T: Hash + Eq + Send + Sync + ToOwned + ?Sized + 'a,
    <T as ToOwned>::Owned: Hash + Eq + Send + Sync + Clone,
{
    /// New instance with default settings.
    pub fn new() -> Self {
        VocabStat::from_setting(Default::default())
    }

    /// New instance with customized settings.
    pub fn from_setting(setting: VocabStatSetting<'a, T>) -> Self {
        Self {
            setting,
            counter: RwLock::new(Default::default()),
            num_processed_sentences: AtomicUsize::new(0),
        }
    }

    /// Update self with a `corpus` of an iterator of strings, each of which is a sequence of words separated by `sep`.
    /// Usually corpus is an `Vec<String>` calling `.par_iter().map(|s| s.as_str())`.
    pub fn update_char_sep_corpus<
        Sep: Send + Sync + Copy,
        S: SplitToWords<Sep, Item = &'a T> + Send,
        I: IntoParallelIterator<Item = S>,
    >(
        self,
        corpus: I,
        sep: Sep,
    ) -> Self {
        {
            let counter = &self.counter;
            let num_processed_sentences = &self.num_processed_sentences;
            corpus.into_par_iter().for_each(|sent| {
                num_processed_sentences.inc();

                let mut new_words = vec![];
                // Stat all in-vocabulary words
                {
                    let rlock = counter.read().unwrap();
                    if self.setting.stat_df {
                        Self::update_counter(
                            sent.split_to_words(sep).collect::<HashSet<_>>(),
                            &rlock,
                            &mut new_words,
                        );
                    } else {
                        Self::update_counter(sent.split_to_words(sep), &rlock, &mut new_words);
                    }
                }

                // Require write lock, recheck and stat all maybe-out-of-vocabulary words
                if !new_words.is_empty() {
                    let mut wlock = counter.write().unwrap();
                    for new_word in new_words {
                        *wlock
                            .entry(new_word)
                            .or_insert(AtomicUsize::new(0))
                            .get_mut() += 1; // get_mut to avoid slow atomic add, since WRITE lock is already acquired
                    }
                }
            });
        }
        self
    }

    fn update_counter<I: IntoIterator<Item = &'a T>>(
        iter: I,
        rlock: &rayon_hash::HashMap<&'a T, AtomicUsize, fnv::FnvBuildHasher>,
        new_words: &mut Vec<&'a T>,
    ) {
        for word in iter {
            if let Some(count) = rlock.get(&word) {
                count.inc();
                continue;
            } else {
                new_words.push(word);
            }
        }
    }

    /// Turn `self` into results: (vocabulary as Vec of words, word to ID mapping)
    pub fn into_vocab_idmap(
        self,
    ) -> (
        Vec<<T as ToOwned>::Owned>,
        HashMap<<T as ToOwned>::Owned, WordId>,
    ) {
        // Get rid of RwLock and Atomic
        let mut vocab = self.counter
            .into_inner()
            .unwrap()
            .into_iter()
            .map(|(word, count)| (word, count.into_inner()))
            .collect::<Vec<_>>();
        // Bitwise reverse for easy decending sort.
        vocab.par_sort_unstable_by_key(|&(_, count)| !count);
        let mut vocab = self.setting
            .extra_markers
            .into_par_iter()
            .chain(vocab.into_par_iter().map(|(word, _)| word))
            .map(|word| word.to_owned())
            .collect::<Vec<_>>();

        if let Some(max_vocab_size) = self.setting.max_vocab_size {
            if vocab.len() > max_vocab_size {
                assert!(max_vocab_size > 0);
                vocab.truncate(max_vocab_size);
            }
        }

        let word_id_map = vocab
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, word)| (word.to_owned(), i as WordId))
            .collect::<HashMap<_, _>>();

        (vocab, word_id_map)
    }

    /// Turn `self` into results: (vocabulary as Vec of words, word to ID mapping, document frequency for each word)
    ///
    /// # Panics
    /// it will panic if `stat_df` in settings is not true.
    pub fn into_vocab_idmap_df(
        self,
    ) -> (
        Vec<<T as ToOwned>::Owned>,
        HashMap<<T as ToOwned>::Owned, WordId>,
        Vec<f32>,
    ) {
        assert!(self.setting.stat_df);

        // Get rid of RwLock and Atomic
        let mut vocab = self.counter
            .into_inner()
            .unwrap()
            .into_iter()
            .map(|(word, count)| (word, count.into_inner()))
            .collect::<Vec<_>>();
        // Bitwise reverse for easy decending sort.
        vocab.par_sort_unstable_by_key(|&(_, count)| !count);
        let num_processed_sentences = self.num_processed_sentences.into_inner();
        let (mut vocab, mut df): (Vec<_>, Vec<_>) = self.setting
            .extra_markers
            .into_par_iter()
            .map(|w| (w.to_owned(), 1.0))
            .chain(vocab.into_par_iter().map(|(word, count)| {
                (
                    word.to_owned(),
                    count as f32 / num_processed_sentences as f32,
                )
            }))
            .unzip();

        if let Some(max_vocab_size) = self.setting.max_vocab_size {
            if vocab.len() > max_vocab_size {
                assert!(max_vocab_size > 0);
                vocab.truncate(max_vocab_size);
                df.truncate(max_vocab_size);
            }
        }

        let word_id_map = vocab
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, word)| (word, i as WordId))
            .collect::<HashMap<_, _>>();

        (vocab, word_id_map, df)
    }
}

/// Settings for vocabulary statistics.
#[derive(Clone, Debug)]
pub struct VocabStatSetting<'a, T: ?Sized + 'a> {
    /// Whether stat document frequency. If true, self.counter is DF of each word,
    /// or else self.counter is sum of TF
    pub stat_df: bool,
    /// Maximum number of vocabulary size, including extra marker words.
    pub max_vocab_size: Option<usize>,
    /// Extra marker words, e.g. PAD, UNK, EOS.
    pub extra_markers: Vec<&'a T>,
}

impl<'a, T: ?Sized> Default for VocabStatSetting<'a, T> {
    fn default() -> Self {
        VocabStatSetting {
            stat_df: false,
            max_vocab_size: None,
            extra_markers: vec![],
        }
    }
}

pub trait SplitToWords<Sep>: Sized {
    type Item;
    type Iter: Iterator<Item = Self::Item>;
    fn split_to_words(self, sep: Sep) -> Self::Iter;
    fn collect_words<C: std::iter::FromIterator<Self::Item>>(self, sep: Sep) -> C {
        self.split_to_words(sep).collect()
    }
}

impl<'a> SplitToWords<char> for &'a str {
    type Item = &'a str;
    type Iter = std::str::SplitTerminator<'a, char>;
    fn split_to_words(self, sep: char) -> Self::Iter {
        self.split_terminator(sep)
    }
}

impl<'a, T> SplitToWords<()> for &'a [T] {
    type Item = &'a T;
    type Iter = std::slice::Iter<'a, T>;
    fn split_to_words(self, _sep: ()) -> Self::Iter {
        self.iter()
    }
}

impl<'a, T> SplitToWords<()> for &'a Vec<T> {
    type Item = &'a T;
    type Iter = std::slice::Iter<'a, T>;
    fn split_to_words(self, _sep: ()) -> Self::Iter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
