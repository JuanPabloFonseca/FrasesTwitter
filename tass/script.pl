#!/usr/bin/perl
#script para juntar los topicos y texto del tweet en una sola linea
#ejecucion perl script.pl
use XML::XPath;

my $file = 'tass_2015_tagged/general-tweets-test.xml'; # general-tweets-train-tagged.xml
my $xp = XML::XPath->new(filename => $file);
my $count = 1;
foreach my $entry ($xp->find('//tweet')->get_nodelist){
	foreach my $topic ($entry->find('topics/topic')->get_nodelist){
	    print $topic->string_value . ",";
	}
	print "\\\\\\";
    print $entry->find('content')->string_value . "\n";
    $count++;
}
