#!/usr/bin/perl
# script para juntar los topicos y texto del tweet en una sola linea
# ejecucion 
# perl -Mutf8 -CS script.pl > TASSTrain.txt
use XML::XPath;

my $file = 'tass_2015_tagged/general-tweets-test1k-tagged.xml'; 
# general-tweets-train-tagged.xml
# general-tweets-test1k-tagged.xml
my $xp = XML::XPath->new(filename => $file);
my $count = 1;
foreach my $entry ($xp->find('//tweet')->get_nodelist){
    print $entry->find('date'); 
    print "\\\\\\\\\\\\";
    print $entry->find('tweetid');
    print "\\\\\\\\\\\\";
    my $str = $entry->find('content')->string_value;
    $str =~ s/\R//g;
    print $str ;
    print "\\\\\\\\\\\\";
    foreach my $topic ($entry->find('topics/topic')->get_nodelist){
	print $topic->string_value . ",";
    }
    print "\n";	

    $count++;
}
